"""Round-29 Stage-2 condition family injection module.

Per analyses/2026-05-26_stage2_cond_injection_ablation_claude_code_prompt.md
§4, this module owns the FIVE injection modes (J0-J4) used to combine
the four condition families (C / I / S / B) into the Stage-2 DiT
residual stream:

    J0 input_add        : one summed input-token projection per family
    J1 gated_input      : J0 + per-family sigmoid gate (bias = -1.0 by default)
    J2 adapter_only     : per-family per-DiT-block adapter, no input add
    J3 input_add+adapter: J0 + per-family per-block adapter
    J4 typed            : per-family choice of {input_add, gated_input, adapter}

Family abbreviations used as ``family_name``:
    "coarse_extra" — C38/C41 EXTRA channel (not Stage-1 23-D itself)
    "interaction"  — I1..I4 hand/object content
    "support"      — S1..S4 foot/gait content
    "body_refine"  — B1..B4 body refinement content

The Stage-1 Coarse-v1 23-D channel is plumbed via the existing
``stage1_coarse`` cond key (separate dedicated zero-init projection in
V12InputProjection), so this module never touches it.

Backward compatibility
----------------------
This module is OFF by default — only activated when an
AnchorDenoiserConfig field ``use_round29_cond_injection=True`` is set,
which is opt-in via the YAML config. R28 oracle_interaction_hint +
body_action_hint paths still work in parallel; the R29 path is a clean,
typed alternative that the R29 ablation matrix uses exclusively.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


FAMILY_NAMES: tuple[str, ...] = (
    "coarse_extra", "interaction", "support", "body_refine",
)


VALID_INJECTION_MODES: tuple[str, ...] = (
    "input_add", "gated_input", "adapter_only", "input_add_adapter", "typed",
)


@dataclass(slots=True)
class Round29CondInjectionConfig:
    """Per-family dim + per-family injection mode.

    Family dim 0 disables that family. Injection mode default is
    uniform across families ('input_add' to 'input_add_adapter') unless
    ``per_family_modes`` is provided (J4 typed-injection).

    Field defaults make a fresh dataclass equivalent to the "all
    families disabled, J0" baseline, so importing this module does not
    change any existing behaviour.
    """
    coarse_extra_dim: int = 0
    interaction_dim: int = 0
    support_dim: int = 0
    body_refine_dim: int = 0
    injection_mode: str = "input_add"
    gate_bias_init: float = -1.0
    # When ``injection_mode == 'typed'`` (J4), per-family mode overrides
    # the global mode. Keys are family names; missing keys fall back to
    # the global ``injection_mode``. Values must be in
    # {'input_add', 'gated_input', 'adapter_only', 'input_add_adapter'}.
    per_family_modes: dict[str, str] | None = None
    zero_init_adapters: bool = True

    def family_dim(self, family: str) -> int:
        return {
            "coarse_extra": int(self.coarse_extra_dim),
            "interaction": int(self.interaction_dim),
            "support": int(self.support_dim),
            "body_refine": int(self.body_refine_dim),
        }[family]

    def effective_mode(self, family: str) -> str:
        if self.injection_mode != "typed":
            return self.injection_mode
        modes = self.per_family_modes or {}
        return modes.get(family, "input_add")

    def active_families(self) -> tuple[str, ...]:
        return tuple(f for f in FAMILY_NAMES if self.family_dim(f) > 0)


class Round29CondInjectionModule(nn.Module):
    """Per-family input projections + optional gates + optional per-layer
    adapters, plus a forward helper that returns the input-token addend
    and a method that returns one per-layer adapter delta.

    Design choices
    --------------
    * Each family has its OWN 2-layer MLP projection (prompt §4.1 forbids
      a monolithic concat MLP). Last-layer Linear is zero-initialised so
      step-0 forward equals the no-condition baseline.
    * For ``adapter_only``, no input projection is performed — the input
      add is identically zero and the family reaches the residual stream
      only via per-block adapters.
    * The gate (for ``gated_input``) is driven by
      ``[c_summary; cond_emb]`` (prompt §4.2) where c_summary is the
      AdaLN global cond vector (B, D); the network sees both the global
      diffusion-time signal and the per-frame condition embedding.
    """

    def __init__(self, cfg: Round29CondInjectionConfig, d_model: int) -> None:
        super().__init__()
        if cfg.injection_mode not in VALID_INJECTION_MODES:
            raise ValueError(
                f"injection_mode must be one of {VALID_INJECTION_MODES}; "
                f"got {cfg.injection_mode!r}"
            )
        if cfg.injection_mode == "typed" and cfg.per_family_modes:
            for f, mode in cfg.per_family_modes.items():
                if f not in FAMILY_NAMES:
                    raise ValueError(
                        f"per_family_modes contains unknown family {f!r}; "
                        f"valid: {FAMILY_NAMES}"
                    )
                if mode not in (
                    "input_add", "gated_input", "adapter_only", "input_add_adapter",
                ):
                    raise ValueError(
                        f"per_family_modes[{f!r}] = {mode!r} is not a base mode"
                    )

        self.cfg = cfg
        self.d_model = int(d_model)

        # Per-family projection MLPs (one per active family).
        self.proj = nn.ModuleDict()
        # Per-family gate Linear (only when family mode is gated_input).
        self.gate = nn.ModuleDict()
        # Per-family per-layer adapter MLPs (only when family mode uses
        # adapters). Built lazily in ``configure_adapter_layers``.
        self.adapters = nn.ModuleDict()
        self._n_layers: int = 0
        self._configured_adapters: bool = False

        for f in cfg.active_families():
            dim = cfg.family_dim(f)
            mode = cfg.effective_mode(f)
            if mode in ("input_add", "gated_input", "input_add_adapter"):
                proj = nn.Sequential(
                    nn.Linear(dim, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                # Zero-init final Linear -> bit-exact identity at step 0.
                nn.init.zeros_(proj[-1].weight)
                nn.init.zeros_(proj[-1].bias)
                self.proj[f] = proj
            elif mode == "adapter_only":
                # adapter_only families still need a shared "cond_emb"
                # representation for per-layer adapters. Use the same
                # 2-layer MLP recipe; output goes ONLY through adapters,
                # never through input add.
                proj = nn.Sequential(
                    nn.Linear(dim, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                # NOTE: leave the projection Xavier-init (default). The
                # zero-init is on the per-block adapter's LAST Linear, so
                # step-0 still matches the no-cond baseline.
                self.proj[f] = proj
            else:
                raise AssertionError(f"unreachable mode {mode!r}")

            if mode == "gated_input":
                gate = nn.Linear(2 * d_model, 1)
                if cfg.zero_init_adapters:
                    nn.init.zeros_(gate.weight)
                    nn.init.constant_(gate.bias, float(cfg.gate_bias_init))
                self.gate[f] = gate

        # Per-forward caches and diagnostics, reset at start of every
        # forward pass on the parent module.
        self._cond_emb_cache: dict[str, Tensor] = {}
        self._last_stats: dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # Configuration of per-layer adapters
    # ------------------------------------------------------------------

    def configure_adapter_layers(self, n_layers: int) -> None:
        """Lazily build per-layer adapters once the DiT block count is known.

        Called by the parent module (AnchorDenoiser) once the
        ``self.v12_blocks`` ModuleList exists; we cannot know ``n_layers``
        at __init__ time without copying it from the parent config.
        """
        if self._configured_adapters:
            return
        self._n_layers = int(n_layers)
        for f in self.cfg.active_families():
            mode = self.cfg.effective_mode(f)
            if mode not in ("adapter_only", "input_add_adapter"):
                continue
            ads = nn.ModuleList()
            for _ in range(self._n_layers):
                adapter = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                if self.cfg.zero_init_adapters:
                    nn.init.zeros_(adapter[-1].weight)
                    nn.init.zeros_(adapter[-1].bias)
                ads.append(adapter)
            self.adapters[f] = ads
        self._configured_adapters = True

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar(t: Tensor) -> Tensor:
        return t.detach().float()

    @staticmethod
    def _mean_norm(x: Tensor) -> Tensor:
        return x.detach().float().norm(dim=-1).mean()

    def _set_stat(self, key: str, value: Tensor) -> None:
        self._last_stats[f"r29_{key}"] = self._scalar(value)

    def last_stats(self) -> dict[str, Tensor]:
        return dict(self._last_stats)

    # ------------------------------------------------------------------
    # Forward — family-embedding compute (PB1 / PB2 prerequisite)
    # ------------------------------------------------------------------

    def compute_family_embeddings(
        self,
        cond: dict[str, Tensor],    # keyed by "stage2_<family>"
    ) -> dict[str, Tensor]:
        """Project each active family's typed cond tensor into the d_model
        embedding space and cache it under ``self._cond_emb_cache``.

        Must be called BEFORE ``apply_input_injection`` (PB1 / PB2 path:
        the parent caches family embeddings once and consumes them from
        both AdaLN pool and the input-add lane). When ``apply_input_injection``
        is called directly (legacy code path), it triggers this internally
        to preserve behaviour.

        Returns the same dict, primarily so callers can introspect/test.
        """
        self._cond_emb_cache = {}
        self._last_stats = {}
        for f in self.cfg.active_families():
            key = f"stage2_{f}"
            if key not in cond:
                raise KeyError(
                    f"Round29CondInjectionModule: cond[{key!r}] is required "
                    f"because family {f!r} is active "
                    f"(dim={self.cfg.family_dim(f)}). The trainer must "
                    f"populate this from the dataset's Stage2ConditionBundle."
                )
            x = cond[key]                                                # (B, T, D_f)
            if x.shape[-1] != self.cfg.family_dim(f):
                raise ValueError(
                    f"cond[{key!r}] last-dim {x.shape[-1]} != configured "
                    f"family dim {self.cfg.family_dim(f)}"
                )
            emb = self.proj[f](x)                                        # (B, T, D)
            self._cond_emb_cache[f] = emb
            self._set_stat(f"{f}_hint_norm", self._mean_norm(x))
            self._set_stat(f"{f}_emb_norm", self._mean_norm(emb))
        return self._cond_emb_cache

    def pool_cond_summary(
        self,
        families: Iterable[str],
        cond: dict[str, Tensor],
        pool: str = "mean",
    ) -> Tensor:
        """Pool cached per-family (B, T, D) embeddings into a single
        per-sample (B, D) vector for AdaLN (PB1).

        Allowed pool modes:
          - ``"mean"``: mean over T, then mean across the requested
            families.
          - ``"support_walking_mean"``: walking_mask-weighted mean of the
            ``support`` family embedding over T. ``walking_mask`` is S4
            dim 4 (see ``configs/training/anchordiff_r29_*`` S4 layout).
            Denominator is clamped to ≥ 1.0 so a clip with zero walking
            frames produces a zero vector instead of NaN. If
            ``support`` is not active or the support family lacks a
            recognisable walking_mask channel (dim < 5), falls back to
            ``mean`` over the available family list and records a
            warning stat.

        Returns (B, D). When ``families`` is empty or no requested family
        is active, returns a zero vector with the right shape inferred
        from the first cached embedding (or raises if nothing is
        cached — that means ``compute_family_embeddings`` was not
        called).
        """
        cache = self._cond_emb_cache
        if not cache:
            raise RuntimeError(
                "pool_cond_summary called before compute_family_embeddings; "
                "AnchorDenoiser.forward must call compute_family_embeddings "
                "first in PB1 path."
            )
        requested = [f for f in families if f in cache]
        if pool == "support_walking_mean":
            if "support" in requested:
                support_emb = cache["support"]                           # (B, T, D)
                support_key = "stage2_support"
                support_x = cond.get(support_key)
                if (
                    support_x is not None
                    and support_x.dim() == 3
                    and support_x.shape[-1] >= 5
                ):
                    w = support_x[..., 4]                                # (B, T)
                    w = w.to(dtype=support_emb.dtype)
                    denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)    # (B, 1)
                    weighted = (support_emb * w.unsqueeze(-1)).sum(dim=1)  # (B, D)
                    pooled = weighted / denom                            # (B, D)
                    self._set_stat(
                        "adaln_support_walking_frac_mean",
                        w.float().mean(),
                    )
                    return pooled
                # Fallback: support active but dim < 5 (no walking_mask
                # channel) — record warning + fall through to mean over
                # support only.
                self._set_stat(
                    "adaln_support_walking_mean_fallback",
                    torch.tensor(1.0, device=support_emb.device),
                )
                return support_emb.mean(dim=1)
            # Fallback: pool mode asked for support but support isn't
            # active. Record warning, fall through to ``mean`` over
            # whatever requested families ARE cached.
            if cache:
                ref = next(iter(cache.values()))
                self._set_stat(
                    "adaln_support_walking_mean_fallback",
                    torch.tensor(1.0, device=ref.device),
                )
        # ``mean`` (or fallback from support_walking_mean): mean over T
        # then mean across families.
        if not requested:
            ref = next(iter(cache.values()))
            return torch.zeros(
                ref.shape[0], ref.shape[-1],
                device=ref.device, dtype=ref.dtype,
            )
        pooled_per_family = [cache[f].mean(dim=1) for f in requested]    # list of (B, D)
        return torch.stack(pooled_per_family, dim=0).mean(dim=0)         # (B, D)

    # ------------------------------------------------------------------
    # Forward — input-add stage
    # ------------------------------------------------------------------

    def apply_input_injection(
        self,
        h: Tensor,                  # (B, T, D)
        cond: dict[str, Tensor],    # keyed by "stage2_<family>"
        c_summary: Tensor | None,
    ) -> Tensor:
        """Compute per-family embeddings (if not already cached), use them
        for per-layer adapters, and return the modified residual stream ``h``.

        For each active family:
            family mode 'input_add'           -> h += proj(cond)
            family mode 'gated_input'         -> h += sigmoid(gate([c; emb])) * emb
            family mode 'input_add_adapter'   -> h += proj(cond)
                                                 (adapter delta added per-block later)
            family mode 'adapter_only'        -> h unchanged (adapter only)

        ``cond`` keys are the TYPED bundle keys:
            cond['stage2_coarse_extra'], cond['stage2_interaction'],
            cond['stage2_support'],     cond['stage2_body_refine']

        Missing keys for active families raise KeyError.

        Cache reuse: when the parent module has already called
        ``compute_family_embeddings(cond)`` earlier this forward pass
        (PB1 / PB2 path), the cache is already populated and we reuse it
        directly — saves one projection pass. When called standalone
        (legacy path), we populate it ourselves.
        """
        active = self.cfg.active_families()
        cache_complete = (
            self._cond_emb_cache and all(f in self._cond_emb_cache for f in active)
        )
        if not cache_complete:
            self.compute_family_embeddings(cond)

        for f in active:
            emb = self._cond_emb_cache[f]

            mode = self.cfg.effective_mode(f)
            if mode == "input_add":
                h = h + emb
            elif mode == "input_add_adapter":
                h = h + emb
            elif mode == "gated_input":
                if c_summary is None:
                    # Allow None — substitute zero vector. Keeps the
                    # module callable from non-DiT entry points (smoke
                    # test) but ConditionedEncoderLayer always provides
                    # one.
                    c_b = torch.zeros(
                        emb.shape[0], emb.shape[-1],
                        device=emb.device, dtype=emb.dtype,
                    )
                else:
                    c_b = c_summary
                gate_in = torch.cat(
                    [c_b.unsqueeze(1).expand(-1, emb.shape[1], -1), emb],
                    dim=-1,
                )                                                        # (B, T, 2D)
                g = torch.sigmoid(self.gate[f](gate_in))                # (B, T, 1)
                self._set_stat(f"{f}_gate_mean", g.mean())
                self._set_stat(f"{f}_gate_std", g.float().std(unbiased=False))
                h = h + g * emb
            elif mode == "adapter_only":
                pass   # no input add; adapter delta added per-block later
            else:
                raise AssertionError(f"unreachable mode {mode!r}")
        return h

    # ------------------------------------------------------------------
    # Forward — per-layer adapter stage
    # ------------------------------------------------------------------

    def apply_per_layer_adapter(
        self,
        seq: Tensor,                # (B, T_total, D) including prefix tokens
        layer_idx: int,
        motion_token_start: int,
    ) -> Tensor:
        """Add per-family adapter deltas to motion tokens of ``seq`` at
        DiT block index ``layer_idx``. No-op when no family uses an
        adapter mode at this layer.

        Adapter delta is ``adapter[l](cond_emb_cache[family])`` and is
        added to ``seq[:, motion_token_start:, :]``. ``cond_emb_cache``
        was populated by ``apply_input_injection`` earlier this forward
        pass.
        """
        added = False
        for f in self.cfg.active_families():
            mode = self.cfg.effective_mode(f)
            if mode not in ("adapter_only", "input_add_adapter"):
                continue
            if f not in self._cond_emb_cache:
                continue
            adapter = self.adapters[f][layer_idx]
            emb = self._cond_emb_cache[f]
            delta = adapter(emb)                                         # (B, T, D)
            self._set_stat(
                f"{f}_adapter_norm_layer{layer_idx}",
                self._mean_norm(delta),
            )
            if not added:
                seq = seq.clone()
                added = True
            seq[:, motion_token_start:, :] = (
                seq[:, motion_token_start:, :] + delta
            )
        return seq


def coerce_per_family_modes(value: object) -> dict[str, str] | None:
    """Helper for config loaders — accept None / dict / DictConfig-like.

    Returns a plain ``dict[str, str]`` for valid input or None.
    """
    if value is None:
        return None
    if hasattr(value, "items"):
        out: dict[str, str] = {}
        for k, v in value.items():
            out[str(k)] = str(v)
        return out
    raise TypeError(
        f"per_family_modes must be None or a mapping; got {type(value).__name__}"
    )


__all__ = [
    "FAMILY_NAMES",
    "VALID_INJECTION_MODES",
    "Round29CondInjectionConfig",
    "Round29CondInjectionModule",
    "coerce_per_family_modes",
]
