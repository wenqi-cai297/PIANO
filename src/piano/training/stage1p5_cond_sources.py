"""Stage-1.5 conditioning source selection for R43 P0 onward.

Stage-1.5's training-time `stage1_coarse` cond used to come exclusively
from GT motion via :func:`extract_coarse_v1_batched` (oracle path).
R42's 2x2 diag showed that this oracle distribution is markedly
different from generated Stage-1 output at deployment, and Stage-1.5
trained on the oracle distribution collapses (drift mean 39 cm, pelvis
drift 33.9 cm) when fed generated Stage-1 in the full cascade.

This module exposes the helper R43 P0 needs: load Stage-1's already-
generated z-scored 23-D coarse from a per-clip ``.npz`` cache produced
by :mod:`piano.inference.sample_substitute_conds`.

Convention (verified against
:mod:`piano.inference.sample_substitute_conds`):

  cache_root/<subset>/<seq_id>.npz
    stage1_coarse: (T_cached, 23) z-scored float32  -- key required
    valid_T:       int32                            -- present, unused here
    seed:          int32                            -- present, unused here

The values are **already z-scored** by Stage-1's training target
convention (``train_stage1.py:411,470``). Do NOT re-normalize.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


_EXPECTED_C = 23


class GeneratedCoarseCacheError(RuntimeError):
    """Raised when the generated cache for a sample is missing or malformed.

    The message always includes (subset, seq_id, full path) so the caller
    can locate the offending entry. The trainer must propagate this; it
    must NOT silently fall back to oracle (Codex r43_p0_finalized §1).
    """


def _cache_path(cache_root: Path, subset: str, seq_id: str) -> Path:
    return cache_root / subset / f"{seq_id}.npz"


def load_generated_coarse_z_for_batch(
    *,
    batch: dict,
    cache_root: Path,
    expected_T: int,
    expected_C: int = _EXPECTED_C,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Load generated z-scored ``stage1_coarse`` for every batch item.

    Lookup uses ``batch["subset"][i]`` and ``batch["seq_id"][i]`` — both
    are list[str] surfaced by ``collate_hoi`` (dataset.py:1332).

    Returns
    -------
    Tensor of shape ``(B, expected_T, expected_C)`` on the requested
    device/dtype. When ``device`` / ``dtype`` are omitted, defaults to
    the batch's ``motion`` tensor's device/dtype so the result drops
    into the existing oracle path without explicit casting.

    Length policy
    -------------
    For each clip:

    * ``T_cached == expected_T`` → use as-is.
    * ``T_cached > expected_T``  → trim to ``[:expected_T]`` (matches the
      inference helper's behavior when the cache holds a longer
      sequence than the current batch view).
    * ``T_cached < expected_T``  → raise :class:`GeneratedCoarseCacheError`
      with ``(subset, seq_id, path, T_cached, expected_T)`` so the
      operator can re-sample the affected clip.

    Validation
    ----------
    Every clip is checked for:

    * file exists
    * key ``stage1_coarse`` present
    * shape ``(T_cached, expected_C)``
    * all finite values

    Any failure raises :class:`GeneratedCoarseCacheError`. The trainer
    NEVER silently falls back to oracle.
    """
    if "subset" not in batch or "seq_id" not in batch:
        raise GeneratedCoarseCacheError(
            "batch is missing 'subset' or 'seq_id' list[str] fields; "
            "ensure dataset.collate_hoi is the collate function so "
            "(dataset.py:1332) string-field preservation kicks in."
        )
    subsets = batch["subset"]
    seq_ids = batch["seq_id"]
    if len(subsets) != len(seq_ids):
        raise GeneratedCoarseCacheError(
            f"batch subset list length {len(subsets)} != seq_id list "
            f"length {len(seq_ids)}"
        )
    B = len(subsets)

    if device is None or dtype is None:
        motion = batch.get("motion")
        if motion is None:
            raise GeneratedCoarseCacheError(
                "batch missing 'motion' tensor; cannot infer device/dtype. "
                "Pass device= and dtype= explicitly."
            )
        if device is None:
            device = motion.device
        if dtype is None:
            dtype = motion.dtype

    out = torch.empty((B, expected_T, expected_C), device=device, dtype=dtype)
    for i in range(B):
        subset = str(subsets[i])
        seq_id = str(seq_ids[i])
        path = _cache_path(cache_root, subset, seq_id)
        if not path.is_file():
            raise GeneratedCoarseCacheError(
                f"missing generated cache entry: subset={subset!r} "
                f"seq_id={seq_id!r} path={path}. Re-run "
                f"sample_substitute_conds for this clip or regenerate "
                f"the full cache."
            )
        try:
            with np.load(path) as data:
                if "stage1_coarse" not in data.files:
                    raise GeneratedCoarseCacheError(
                        f"cache entry missing 'stage1_coarse' key: "
                        f"subset={subset!r} seq_id={seq_id!r} path={path} "
                        f"keys={list(data.files)}"
                    )
                arr = np.asarray(data["stage1_coarse"], dtype=np.float32)
        except GeneratedCoarseCacheError:
            raise
        except Exception as exc:
            raise GeneratedCoarseCacheError(
                f"failed to read npz: subset={subset!r} seq_id={seq_id!r} "
                f"path={path} error={exc!r}"
            ) from exc

        if arr.ndim != 2 or arr.shape[1] != expected_C:
            raise GeneratedCoarseCacheError(
                f"unexpected stage1_coarse shape {arr.shape}; expected "
                f"(T, {expected_C}). subset={subset!r} seq_id={seq_id!r} "
                f"path={path}"
            )
        T_cached = arr.shape[0]
        if T_cached < expected_T:
            raise GeneratedCoarseCacheError(
                f"cached T={T_cached} < expected T={expected_T}; "
                f"subset={subset!r} seq_id={seq_id!r} path={path}. "
                f"Re-sample this clip with a long-enough horizon."
            )
        arr = arr[:expected_T]
        if not np.isfinite(arr).all():
            raise GeneratedCoarseCacheError(
                f"non-finite values in cached stage1_coarse: "
                f"subset={subset!r} seq_id={seq_id!r} path={path}"
            )
        out[i] = torch.from_numpy(arr).to(device=device, dtype=dtype)

    return out


def select_stage1_coarse(
    *,
    cond_source: str,
    oracle_z: torch.Tensor,
    batch: dict,
    cache_root: Path | None,
    generated_prob: float,
    training: bool,
) -> torch.Tensor:
    """Select the Stage-1 coarse cond for one training/eval step.

    Parameters
    ----------
    cond_source : ``"oracle"`` | ``"generated_cache"`` | ``"mixed"``
    oracle_z : (B, T, 23) — already z-scored, from
        ``extract_coarse_v1_batched`` + ``(x - mean) / std`` in trainer.
    batch : passed through to the cache loader for subset/seq_id/motion.
    cache_root : required when ``cond_source != "oracle"``.
    generated_prob : per-sample probability of picking generated under
        ``"mixed"`` mode at training time. Ignored otherwise.
    training : when ``True``, mixed mode samples per-item; when ``False``
        (eval / val loss), mixed mode uses generated only — matches the
        deployment metric (Codex r43_p0_finalized_review §3.2).

    Returns
    -------
    Same shape/device/dtype as ``oracle_z``. Mixed mode never re-orders
    the batch — the per-item Bernoulli mask is broadcast over (T, 23).
    """
    if cond_source == "oracle":
        return oracle_z

    if cache_root is None:
        raise ValueError(
            f"cond_source={cond_source!r} requires cache_root to be set; "
            "got None."
        )

    B, T, C = oracle_z.shape
    gen_z = load_generated_coarse_z_for_batch(
        batch=batch,
        cache_root=cache_root,
        expected_T=T,
        expected_C=C,
        device=oracle_z.device,
        dtype=oracle_z.dtype,
    )

    if cond_source == "generated_cache":
        return gen_z

    if cond_source == "mixed":
        if not training:
            return gen_z
        if not (0.0 <= generated_prob <= 1.0):
            raise ValueError(
                f"generated_prob must lie in [0, 1]; got {generated_prob!r}"
            )
        use_gen = (
            torch.rand(B, device=oracle_z.device) < float(generated_prob)
        )
        return torch.where(use_gen[:, None, None], gen_z, oracle_z)

    raise ValueError(
        f"unknown cond_source={cond_source!r}; must be 'oracle', "
        "'generated_cache', or 'mixed'."
    )


VALID_COND_SOURCES = ("oracle", "generated_cache", "mixed")
