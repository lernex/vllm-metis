# SPDX-License-Identifier: Apache-2.0
"""Metis-1.6 (Logos and Praxis) for vLLM.

Semantics follow the Hugging Face port (``modeling_metis16.py``), which is the reference: every
token runs the whole layer stack ``num_passes`` times with shared weights, reads and writes four
residual streams through pass-conditioned manifold hyper-connections, runs a Mamba-2 or a
position-free GQA mixer, then a latent mixture of experts, and keeps a small per-token depth
memory that it writes after every attention layer and at the end of every pass.

Cache design
------------
Only the mixers look across positions, so the decode state is one Mamba state or one KV cache
per (layer, pass). vLLM's hybrid allocator gives a cache slot to every attention or Mamba layer
object registered in the forward context under its own name, so each mixer builds one such
object per pass:

* attention layers hold ``attn_pass0`` and ``attn_pass1``, two ``Attention`` objects with no
  weights of their own, called with the same QKV and output projections;
* Mamba layers hold ``ssm_pass0`` (a ``MambaMixer2`` that owns the weights) and ``ssm_pass1``
  (built on the meta device, then pointed at every parameter-bearing submodule of the first).

Logos thus has 6 KV caches and 34 Mamba states, and every weight is loaded once. Because each
slot is an ordinary vLLM layer, chunked prefill, prefix caching, CUDA graphs and the fused
Triton kernels all work unchanged.

Numerics
--------
FP32 is kept where the reference keeps it: routers and the balance bias, ``A``/``D``/``dt_bias``,
the SSM state in the cache (``mamba_ssm_cache_dtype`` defaults to float32 for this model), the
Sinkhorn projection, the depth-memory attention and gates, the continuation head, every RMSNorm
(weight applied before the cast back, as the reference does), and Mamba's grouped gated norm.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import _encode_layer_name

# Checkpoint tensors that belong to the adaptive-depth and adaptive-width heads. The released
# weights were trained under one fixed policy (two passes, four experts), these heads never
# received a gradient under it, and the reference never reads them.
UNUSED_CHECKPOINT_SUFFIXES = (
    ".k_router.weight",
    ".k_router.bias",
    ".k_budget.multiplier",
)
UNUSED_CHECKPOINT_NAMES = ("depth_budget.multiplier",)


# ---------------------------------------------------------------------------- numerics


def _rms_norm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in FP32 with the weight applied before the cast back, as the reference does."""
    wide = values.float()
    wide = wide * torch.rsqrt(wide.square().mean(dim=-1, keepdim=True) + eps)
    return (wide * weight.float()).to(values.dtype)


def _gated_group_rms_norm(
    values: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
) -> torch.Tensor:
    """Mamba-2's gated norm, ``norm(y * silu(z))`` per group of channels, all in FP32."""
    wide = values.float() * F.silu(gate.float())
    grouped = wide.view(*wide.shape[:-1], -1, group_size)
    grouped = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    return (grouped.view_as(wide) * weight.float()).to(values.dtype)


def _sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    log_matrix = logits.float()
    for _ in range(iterations):
        log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)
        log_matrix = log_matrix - torch.logsumexp(log_matrix, dim=-2, keepdim=True)
    return log_matrix.exp()


def _fp32_linear(module: nn.Linear, values: torch.Tensor) -> torch.Tensor:
    bias = module.bias.float() if module.bias is not None else None
    return F.linear(values.float(), module.weight.float(), bias)


def metis_route(
    logits: torch.Tensor,
    selection_bias: torch.Tensor,
    top_k: int,
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert choice as the reference makes it.

    The balance bias moves which experts are chosen, never how they are weighted: pick
    ``topk(logits + bias, max_k)[:top_k]``, then softmax over the chosen unbiased logits.
    """
    logits = logits.float()
    chosen = torch.topk(logits + selection_bias.float(), k=max_k, dim=-1).indices[..., :top_k]
    weights = torch.softmax(logits.gather(-1, chosen), dim=-1)
    return chosen, weights


# Quantizing the Mamba output projection costs more than everything else together: it is 2% of
# Logos' weights and nearly all of an FP8 or 4-bit model's error (top-20 KL to the reference 0.02 to
# 0.09 at FP8 per tensor, against 0.002 to 0.008 for BF16). Online quantization leaves it in the
# model dtype unless the user's own ignore list already mentions it.
MAMBA_OUT_PROJ = "*.ssm_pass*.out_proj"


def _keep_mamba_out_proj_unquantized(quant_config: QuantizationConfig | None) -> None:
    ignored = getattr(quant_config, "ignored_layers", None)
    if not isinstance(ignored, list) or any("out_proj" in pattern for pattern in ignored):
        return
    ignored.append(MAMBA_OUT_PROJ)


class _MetisRouting:
    """The FusedMoE custom routing function. Reads the bias off the MoE module at call time."""

    def __init__(self, moe: "MetisMoE") -> None:
        self._moe = [moe]  # a list, so the MoE is not registered as a child of its own router

    def __call__(self, hidden_states, gating_output, topk, renormalize, **_):
        moe = self._moe[0]
        chosen, weights = metis_route(gating_output, moe.selection_bias, topk, moe.max_k)
        return weights, chosen.to(torch.int32)


# ---------------------------------------------------------------------------- modules


class MetisRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return _rms_norm(values, self.weight, self.eps)


class MetisHyperConnection(nn.Module):
    """Four-stream manifold hyper-connection.

    The 4x4 doubly stochastic mix and both weight vectors depend only on the pass, never on the
    token, so they are computed once after the weights load (``finalize``) and kept as buffers.
    """

    def __init__(self, config, num_passes: int) -> None:
        super().__init__()
        n = config.n_streams
        self.n_streams = n
        self.iterations = config.mhc_sinkhorn_iterations
        self.norm = MetisRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.read_logits = nn.Parameter(torch.zeros(n))
        self.mix_logits = nn.Parameter(torch.zeros(n, n))
        self.write_logits = nn.Parameter(torch.zeros(n))
        self.controller = nn.Linear(config.mhc_pass_embedding_dim, n * n + 2 * n, bias=True)
        self.register_buffer("mix", torch.zeros(num_passes, n, n), persistent=False)
        self.register_buffer("read_weights", torch.zeros(num_passes, n), persistent=False)
        self.register_buffer("write_weights", torch.zeros(num_passes, n), persistent=False)

    @torch.no_grad()
    def finalize(self, pass_embeddings: torch.Tensor) -> None:
        n = self.n_streams
        for p in range(self.mix.shape[0]):
            # Same op order and dtypes as the reference: the controller runs in the weight dtype.
            delta = self.controller(pass_embeddings[p]).float()
            matrix_delta, read_delta, write_delta = torch.split(delta, (n * n, n, n), dim=-1)
            matrix = _sinkhorn(self.mix_logits.float() + matrix_delta.view(n, n), self.iterations)
            self.mix[p].copy_(matrix)
            self.read_weights[p].copy_(torch.softmax(self.read_logits.float() + read_delta, -1))
            self.write_weights[p].copy_(torch.softmax(self.write_logits.float() + write_delta, -1))

    def read(self, streams: torch.Tensor, pass_index: int):
        mixed = torch.einsum("oi,nid->nod", self.mix[pass_index].to(streams.dtype), streams)
        source = torch.einsum("i,nid->nd", self.read_weights[pass_index].to(streams.dtype), streams)
        return self.norm(source), mixed

    def write(self, mixed: torch.Tensor, update: torch.Tensor, pass_index: int) -> torch.Tensor:
        weights = self.write_weights[pass_index].to(update.dtype)
        return mixed + torch.einsum("i,nd->nid", weights, update)


class MetisMambaMixer(MambaMixer2):
    """One (layer, pass) Mamba-2 cache slot.

    The first pass owns the weights. Later passes are built on the meta device, so they allocate
    nothing, and then point every parameter-bearing submodule at the first pass. Each instance
    registers its own prefix in the forward context, so vLLM gives it its own conv and SSM state
    and its own metadata entry; the conv/SSM path is MambaMixer2's own, unchanged.
    """

    def __init__(
        self,
        config,
        *,
        model_config: ModelConfig | None,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        owner: "MetisMambaMixer | None" = None,
    ) -> None:
        intermediate = config.hidden_size * config.mamba_expand
        kwargs = dict(
            hidden_size=config.hidden_size,
            ssm_state_size=config.mamba_d_state,
            conv_kernel_size=config.mamba_d_conv,
            intermediate_size=intermediate,
            use_conv_bias=True,
            use_bias=False,
            n_groups=config.mamba_n_groups,
            num_heads=intermediate // config.mamba_head_dim,
            head_dim=config.mamba_head_dim,
            rms_norm_eps=config.mamba_norm_eps,
            activation="silu",
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        if owner is None:
            super().__init__(**kwargs)
            # The checkpoint and the reference keep D and dt_bias in FP32; MambaMixer2 would
            # create them in the model dtype.
            heads = self.num_heads // self.tp_size
            self.D = nn.Parameter(torch.ones(heads, dtype=torch.float32))
            self.dt_bias = nn.Parameter(torch.ones(heads, dtype=torch.float32))
            set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
            set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        else:
            device = owner.A.device
            with torch.device("meta"):
                super().__init__(**kwargs)
            for name in ("in_proj", "conv1d", "out_proj", "norm"):
                setattr(self, name, getattr(owner, name))
            for name in ("A", "D", "dt_bias"):
                setattr(self, name, getattr(owner, name))
            for name, buffer in owner.named_buffers(recurse=False):
                self.register_buffer(name, buffer, persistent=False)
            # Tensors MambaMixer2 created outside parameters and buffers: give them real storage.
            self.kv_cache = tuple(torch.tensor([], device=device) for _ in self.kv_cache)
            self._replayssm_ring_start = torch.empty(0, dtype=torch.int32, device=device)
            self._replayssm_prev_num_accepted = torch.empty(0, dtype=torch.int32, device=device)
        self.group_size = intermediate // config.mamba_n_groups
        self.norm_eps = config.mamba_norm_eps

    def forward(self, hidden_states: torch.Tensor, mup_vector: torch.Tensor | None = None):
        projected_states, _ = self.in_proj(hidden_states)
        ssm_output = torch.empty(
            (hidden_states.shape[0], self.tped_intermediate_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.mamba_mixer2(projected_states, ssm_output, _encode_layer_name(self.prefix))
        gate = projected_states[..., : self.tped_intermediate_size]
        normed = _gated_group_rms_norm(
            ssm_output, gate, self.norm.weight, self.group_size, self.norm_eps
        )
        output, _ = self.out_proj(normed)
        return output


class MetisMambaLayer(nn.Module):
    def __init__(self, config, num_passes, *, model_config, cache_config, quant_config, prefix):
        super().__init__()
        first = MetisMambaMixer(
            config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.ssm_pass0",
        )
        passes = [first]
        for p in range(1, num_passes):
            passes.append(
                MetisMambaMixer(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.ssm_pass{p}",
                    owner=first,
                )
            )
        self.passes = nn.ModuleList(passes)

    def forward(self, hidden_states: torch.Tensor, pass_index: int) -> torch.Tensor:
        return self.passes[pass_index](hidden_states)


class MetisAttention(nn.Module):
    """Position-free GQA with a per-pass low-rank adapter on QKV, one KV cache per pass."""

    def __init__(self, config, num_passes, *, cache_config, quant_config, prefix):
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            return_bias=False,
        )
        self.out = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out",
            return_bias=False,
        )
        # The adapter stays in the model dtype; it is rank 8 and costs nothing.
        self.pass_lora_down = ReplicatedLinear(
            config.hidden_size,
            config.attention_pass_lora_rank,
            bias=False,
            prefix=f"{prefix}.pass_lora_down",
            return_bias=False,
        )
        self.pass_lora_up = QKVParallelLinear(
            config.attention_pass_lora_rank,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            prefix=f"{prefix}.pass_lora_up",
            return_bias=False,
        )
        self.pass_gates = nn.Parameter(torch.zeros(config.max_passes))
        self.register_buffer("gate_values", torch.zeros(config.max_passes), persistent=False)
        self.attn = nn.ModuleList(
            Attention(
                self.num_heads,
                self.head_dim,
                self.head_dim**-0.5,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn_pass{p}",
            )
            for p in range(num_passes)
        )

    @torch.no_grad()
    def finalize(self) -> None:
        self.gate_values.copy_(torch.tanh(self.pass_gates))

    def forward(self, hidden_states: torch.Tensor, pass_index: int) -> torch.Tensor:
        adapter = self.pass_lora_up(self.pass_lora_down(hidden_states))
        gate = self.gate_values[pass_index].to(hidden_states.dtype)
        qkv = self.qkv(hidden_states) + gate * adapter
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return self.out(self.attn[pass_index](q, k, v))


class MetisSharedExpert(nn.Module):
    def __init__(self, latent: int, intermediate: int, prefix: str) -> None:
        super().__init__()
        self.gate_up = ReplicatedLinear(
            latent, 2 * intermediate, bias=False, prefix=f"{prefix}.gate_up", return_bias=False
        )
        self.down = ReplicatedLinear(
            intermediate, latent, bias=False, prefix=f"{prefix}.down", return_bias=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(hidden).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class MetisMoE(nn.Module):
    """Latent MoE: route on (latent, routing features), top-k of the fused experts plus a shared one."""

    def __init__(self, config, *, quant_config, prefix) -> None:
        super().__init__()
        latent, route = config.latent_dim, config.route_feature_dim
        self.top_k = config.num_experts_per_tok
        self.max_k = config.max_routed_k
        self.latent_down = ReplicatedLinear(
            config.hidden_size, latent, bias=False, prefix=f"{prefix}.latent_down", return_bias=False
        )
        self.latent_up = ReplicatedLinear(
            latent, config.hidden_size, bias=False, prefix=f"{prefix}.latent_up", return_bias=False
        )
        self.expert_router = nn.Linear(
            latent + route, config.n_routed_experts, bias=True, dtype=torch.float32
        )
        self.selection_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32), requires_grad=False
        )
        self.expert_embeddings = nn.Parameter(torch.empty(config.n_routed_experts, route))
        self.shared_expert = MetisSharedExpert(
            latent, config.shared_expert_intermediate_size, f"{prefix}.shared_expert"
        )
        self.experts = FusedMoEFactory(
            num_experts=config.n_routed_experts,
            top_k=self.top_k,
            hidden_size=latent,
            intermediate_size=config.expert_intermediate_size,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=_MetisRouting(self),
            scoring_func="softmax",
            activation="silu",
            router_logits_dtype=torch.float32,
        )

    def forward(self, hidden_states: torch.Tensor, route_features: torch.Tensor):
        latent = self.latent_down(hidden_states)
        route_input = torch.cat((latent, route_features.to(latent.dtype)), dim=-1)
        logits = _fp32_linear(self.expert_router, route_input)
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-9).log()).sum(dim=-1)
        confidence = probabilities.amax(dim=-1)
        # The fused layer recomputes this same choice from the same logits inside its router.
        chosen, weights = metis_route(logits, self.selection_bias, self.top_k, self.max_k)
        routed = self.experts(hidden_states=latent, router_logits=logits)
        output = self.latent_up(routed + self.shared_expert(latent))
        embeddings = self.expert_embeddings
        summary = (weights.to(embeddings.dtype).unsqueeze(-1) * embeddings[chosen]).sum(dim=-2)
        return output, (summary, entropy, confidence)


class MetisDepthMemory(nn.Module):
    """The per-token memory of its own trajectory through the passes.

    Keys and values are projected once per entry, when it is written, instead of re-projecting
    the whole bank on every read; the math is the same.
    """

    def __init__(self, config) -> None:
        super().__init__()
        d, m, r = config.hidden_size, config.memory_dim, config.route_feature_dim
        self.hidden_size, self.memory_dim = d, m
        self.max_k_fraction = float(config.num_experts_per_tok) / float(config.max_routed_k)
        self.entropy_normalizer = (
            math.log(config.n_routed_experts) if config.n_routed_experts > 1 else 1.0
        )
        self.state_write = nn.Linear(d, m, bias=True)
        self.metadata_write = nn.Linear(r + 4, m, bias=True)
        self.pass_embeddings = nn.Parameter(torch.empty(config.max_passes, m))
        self.anchor_embeddings = nn.Parameter(
            torch.empty(len(config.attention_layer_indices) + 1, m)
        )
        self.query = nn.Linear(d, m, bias=True)
        self.key = nn.Linear(m, m, bias=True)
        self.value = nn.Linear(m, m, bias=True)
        self.output = nn.Linear(m, d, bias=True)
        self.route_projection = nn.Linear(3 * d + r, r, bias=True)
        self.stream_gate = nn.Parameter(torch.zeros(config.n_streams, d))
        self.stream_gate_bias = nn.Parameter(torch.zeros(config.n_streams))

    def features(self, state, retrieved, difference, history):
        joined = torch.cat((state, retrieved, difference, history), dim=-1)
        return torch.tanh(self.route_projection(joined))

    def retrieve(self, keys: list[torch.Tensor], values: list[torch.Tensor], streams):
        if not keys:
            return streams, streams.new_zeros(streams.shape[0], self.hidden_size)
        key = torch.stack(keys, dim=1)  # [N, m, M]
        value = torch.stack(values, dim=1)
        query = self.query(streams)  # [N, S, M]
        scores = torch.einsum("nsh,nmh->nsm", query.float(), key.float())
        weights = torch.softmax(scores / math.sqrt(self.memory_dim), dim=-1)
        retrieved = torch.einsum("nsm,nmh->nsh", weights, value.float()).to(value.dtype)
        projected = self.output(retrieved)
        gate_logits = (streams.float() * self.stream_gate.float()).sum(dim=-1)
        gate_logits = gate_logits / math.sqrt(self.hidden_size)
        gates = torch.sigmoid(gate_logits + self.stream_gate_bias.float())
        fused = streams + gates[..., None].to(streams.dtype) * projected
        return fused, projected.mean(dim=1)

    def read(self, keys, values, streams, difference, history):
        streams, summary = self.retrieve(keys, values, streams)
        return streams, self.features(streams.mean(dim=1), summary, difference, history)

    def write(self, keys, values, streams, route, continuation, pass_index, anchor_index):
        summary, entropy, confidence = route
        state = streams.mean(dim=1)
        mean_k = torch.full_like(entropy, self.max_k_fraction)
        metadata = torch.cat(
            (
                summary.float(),
                mean_k.unsqueeze(-1),
                (entropy / self.entropy_normalizer).unsqueeze(-1),
                confidence.unsqueeze(-1),
                continuation.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        entry = (
            self.state_write(state)
            + self.metadata_write(metadata.to(state.dtype))
            + self.pass_embeddings[pass_index]
            + self.anchor_embeddings[anchor_index]
        )
        keys.append(self.key(entry))
        values.append(self.value(entry))


class MetisContinuation(nn.Module):
    """The depth head. Under the fixed two-pass policy its output is only a memory feature."""

    def __init__(self, config) -> None:
        super().__init__()
        d, r = config.hidden_size, config.route_feature_dim
        self.limit = config.continuation_logit_limit
        self.hidden = nn.Linear(3 * d + r, r, bias=True, dtype=torch.float32)
        self.output = nn.Linear(r, 1, bias=True, dtype=torch.float32)

    def forward(self, state, retrieved, difference, route_features):
        features = torch.cat((state, retrieved, difference, route_features), dim=-1)
        hidden = F.silu(_fp32_linear(self.hidden, features))
        logits = _fp32_linear(self.output, hidden).squeeze(-1)
        scale = state.float().square().mean(dim=-1).sqrt().clamp_min(1.0e-6)
        innovation = difference.float().square().mean(dim=-1).sqrt()
        logits = logits + torch.tanh(innovation / scale) - 0.5
        return torch.sigmoid(logits.clamp(-self.limit, self.limit))


class MetisBlock(nn.Module):
    def __init__(self, config, layer_index, num_passes, *, model_config, cache_config,
                 quant_config, prefix) -> None:
        super().__init__()
        self.is_attention = layer_index in config.attention_layer_indices
        if self.is_attention:
            self.mixer = MetisAttention(
                config, num_passes, cache_config=cache_config, quant_config=quant_config,
                prefix=f"{prefix}.mixer",
            )
        else:
            self.mixer = MetisMambaLayer(
                config, num_passes, model_config=model_config, cache_config=cache_config,
                quant_config=quant_config, prefix=f"{prefix}.mixer",
            )
        self.mixer_connection = MetisHyperConnection(config, num_passes)
        self.moe = MetisMoE(config, quant_config=quant_config, prefix=f"{prefix}.moe")
        self.moe_connection = MetisHyperConnection(config, num_passes)

    def forward(self, streams, route_features, pass_index: int):
        mixer_input, mixed = self.mixer_connection.read(streams, pass_index)
        streams = self.mixer_connection.write(mixed, self.mixer(mixer_input, pass_index), pass_index)
        moe_input, mixed = self.moe_connection.read(streams, pass_index)
        moe_output, route = self.moe(moe_input, route_features)
        return self.moe_connection.write(mixed, moe_output, pass_index), route


# ---------------------------------------------------------------------------- model


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Metis16Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        _keep_mamba_out_proj_unquantized(quant_config)
        self.config = config
        self.num_passes = config.num_passes
        self.embedding = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embedding"
        )
        self.stream_embeddings = nn.Parameter(torch.empty(config.n_streams, config.hidden_size))
        self.pass_embeddings = nn.Parameter(
            torch.empty(config.max_passes, config.mhc_pass_embedding_dim)
        )
        self.depth_memory = MetisDepthMemory(config)
        self.layers = nn.ModuleList(
            MetisBlock(
                config,
                index,
                self.num_passes,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{index}",
            )
            for index in range(config.num_hidden_layers)
        )
        self.continuation = MetisContinuation(config)
        self.final_norm = MetisRMSNorm(config.hidden_size, config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    @torch.no_grad()
    def finalize(self) -> None:
        for layer in self.layers:
            layer.mixer_connection.finalize(self.pass_embeddings)
            layer.moe_connection.finalize(self.pass_embeddings)
            if layer.is_attention:
                layer.mixer.finalize()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeds = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        memory = self.depth_memory
        streams = embeds.unsqueeze(1) + self.stream_embeddings.to(embeds.dtype)  # [N, S, D]
        previous_state = streams.mean(dim=1)
        route_history = embeds.new_zeros(embeds.shape[0], self.config.route_feature_dim)
        continuation = embeds.new_zeros(embeds.shape[0])
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        # Inside a pass the reference reads memory with difference = mean(streams) - the state the
        # last pass ended on, taken before any layer runs: always exactly zero.
        no_difference = torch.zeros_like(previous_state)

        for pass_index in range(self.num_passes):
            anchor = 0
            route = None
            for block in self.layers:
                streams, features = memory.read(keys, values, streams, no_difference, route_history)
                streams, route = block(streams, features, pass_index)
                route_history = 0.5 * route_history + 0.5 * route[0]
                if block.is_attention:
                    memory.write(keys, values, streams, route, continuation, pass_index, anchor)
                    anchor += 1
            streams, summary = memory.retrieve(keys, values, streams)
            if pass_index + 1 == self.num_passes:
                # The last pass's depth head and memory entry are never read again.
                break
            current_state = streams.mean(dim=1)
            difference = current_state - previous_state
            features = memory.features(current_state, summary, difference, route_history)
            continuation = self.continuation(current_state, summary, difference, features)
            continuation = continuation.to(streams.dtype)
            memory.write(keys, values, streams, route, continuation, pass_index, anchor)
            previous_state = current_state

        return self.final_norm(streams.mean(dim=1))


class Metis16ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsMambaPrefixCaching):
    """vLLM entry point for ``Metis16ForCausalLM`` checkpoints (Metis-1.6 Logos and Praxis)."""

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, ...]:
        cache_config = vllm_config.cache_config
        base = MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            cache_config.mamba_cache_dtype,
            cache_config.mamba_ssm_cache_dtype,
        )
        if getattr(cache_config, "use_replayssm", False):
            return MambaStateDtypeCalculator.append_replayssm_ring(
                base, vllm_config.model_config.dtype
            )
        return base

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        tp = vllm_config.parallel_config.tensor_parallel_size
        intermediate = config.hidden_size * config.mamba_expand
        base = MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate,
            tp_world_size=tp,
            n_groups=config.mamba_n_groups,
            num_heads=intermediate // config.mamba_head_dim,
            head_dim=config.mamba_head_dim,
            state_size=config.mamba_d_state,
            conv_kernel=config.mamba_d_conv,
            num_spec=vllm_config.num_speculative_tokens,
        )
        if getattr(cache_config, "use_replayssm", False):
            return MambaStateShapeCalculator.append_replayssm_ring(
                base_shapes=base,
                n_groups=config.mamba_n_groups,
                tp_world_size=tp,
                logical_window=cache_config.replayssm_buffer_len,
                backend=vllm_config.mamba_config.backend,
            )
        return base

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.mamba2_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                "Metis-1.6 runs every layer twice per token, so it cannot be split into "
                "pipeline stages; use tensor parallelism instead."
            )
        self.config = config
        self.model = Metis16Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head = self.lm_head.tie_weights(self.model.embedding)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def process_weights_after_loading(self) -> None:
        self.model.finalize()

    # ------------------------------------------------------------------ weights

    def map_checkpoint_name(self, name: str) -> tuple[str, ...] | None:
        """Where a checkpoint tensor goes: ``None`` to skip it, else the vLLM parameter name(s).

        Expert stacks map to the fused layer's ``w13``/``w2`` and are split per expert by
        ``load_weights``. Everything else maps one to one.
        """
        if name in UNUSED_CHECKPOINT_NAMES or name.endswith(UNUSED_CHECKPOINT_SUFFIXES):
            return None
        if name == "lm_head.weight":
            if getattr(self.config, "tie_word_embeddings", True):
                return None
            return ("lm_head.weight",)
        name = "model." + name
        if ".mixer.impl.mixer." in name:
            name = name.replace(".mixer.impl.mixer.", ".mixer.passes.0.")
            if name.endswith(".A_log"):
                name = name[: -len("A_log")] + "A"
        elif ".moe.local_experts.gate_up.weight" in name:
            name = name.replace(".local_experts.gate_up.weight", ".experts.routed_experts.w13_weight")
        elif ".moe.local_experts.down.weight" in name:
            name = name.replace(".local_experts.down.weight", ".experts.routed_experts.w2_weight")
        return (name,)

    def _expert_shards(
        self, param_name: str, stacked: torch.Tensor
    ) -> Iterator[tuple[torch.Tensor, str, int]]:
        # In an FP8 checkpoint the per-channel scales are stacked the same way as the weights.
        if param_name.endswith(("w13_weight", "w13_weight_scale")):
            half = stacked.shape[1] // 2
            for expert in range(stacked.shape[0]):
                yield stacked[expert, :half], "w1", expert
                yield stacked[expert, half:], "w3", expert
        else:
            for expert in range(stacked.shape[0]):
                yield stacked[expert], "w2", expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for checkpoint_name, tensor in weights:
            mapped = self.map_checkpoint_name(checkpoint_name)
            if mapped is None:
                continue
            (name,) = mapped
            if name not in params:
                raise KeyError(f"{checkpoint_name} maps to {name}, which this model does not have")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if name.endswith(("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")):
                for shard, shard_id, expert in self._expert_shards(name, tensor):
                    loader(param, shard, name, shard_id=shard_id, expert_id=expert)
            else:
                loader(param, tensor)
            loaded.add(name)
        return loaded
