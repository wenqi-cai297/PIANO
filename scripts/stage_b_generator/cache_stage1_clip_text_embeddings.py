"""Round-12 preflight (Step 4): cache pooled CLIP text embeddings.

Builds a deduped mapping from clip text to pooled CLIP EOT embedding
(``clip_model.encode_text(tokens).float()``) for the Stage-1 cache.

Why pooled and not mean-pooled per-token:
    The Codex review (analyses/2026-05-21_codex_stage1_coarse_prior_design_review.md §6.1)
    explicitly says not to approximate pooled CLIP by mean-pooling
    per-token features unless equivalence is validated. OpenAI CLIP
    pools via the EOT-token projection inside ``encode_text``; we use
    that path directly here.

Stored fields
-------------

``text_embeddings_clip_vit_b32.npz`` with:

- ``texts``     : list[str] of length N_unique (saved as a 1-D object array)
- ``embeddings``: (N_unique, 512) float32 pooled CLIP text features
- ``clip_model_name``: str
- ``download_root``: str (where the CLIP weights came from)
- ``index``     : dict mapping raw clip text -> row index (saved as JSON sibling)

The trainer maps each manifest record's `text` field through the
``index`` dict to look up the embedding row.

Usage
-----

    $env:PYTHONIOENCODING="utf-8"
    conda run -n piano python scripts/stage_b_generator/cache_stage1_clip_text_embeddings.py \
        --cache-root cache/stage1_coarse_v1_round12 \
        --clip-version "ViT-B/32" \
        --clip-download-root cache/clip
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from piano.utils.clip_utils import load_clip_text_encoder, set_clip_cache_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("cache/stage1_coarse_v1_round12"))
    parser.add_argument("--clip-version", type=str, default="ViT-B/32")
    parser.add_argument("--clip-download-root", type=str, default="cache/clip")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cache_root: Path = args.cache_root
    if not cache_root.exists():
        raise SystemExit(f"[clip] cache root {cache_root} missing — build the Coarse-v1 cache first.")

    texts: list[str] = []
    for name in ("manifest_train.jsonl", "manifest_val.jsonl"):
        for line in (cache_root / name).read_text("utf-8").splitlines():
            if line.strip():
                texts.append(json.loads(line).get("text", ""))
    unique_texts = sorted(set(texts))
    print(f"[clip] total clips = {len(texts)}   unique texts = {len(unique_texts)}")

    set_clip_cache_root(args.clip_download_root)
    device = torch.device(args.device)
    model = load_clip_text_encoder(
        device=device,
        model_name=args.clip_version,
        download_root=args.clip_download_root,
    )

    import clip

    embeddings = np.zeros((len(unique_texts), 512), dtype=np.float32)
    t_start = time.time()
    for i0 in range(0, len(unique_texts), args.batch_size):
        chunk = unique_texts[i0 : i0 + args.batch_size]
        with torch.no_grad():
            tokens = clip.tokenize(chunk, truncate=True).to(device)
            pooled = model.encode_text(tokens).float().cpu().numpy()
        embeddings[i0 : i0 + len(chunk)] = pooled
    elapsed = time.time() - t_start
    print(f"[clip] encoded {len(unique_texts)} unique texts in {elapsed:.1f}s on {device}")

    # Finite check
    if not np.isfinite(embeddings).all():
        raise SystemExit("[clip] non-finite embedding produced — aborting.")

    index = {t: i for i, t in enumerate(unique_texts)}

    # Save
    np.savez_compressed(
        cache_root / "text_embeddings_clip_vit_b32.npz",
        embeddings=embeddings,
        texts=np.asarray(unique_texts, dtype=object),
        clip_model_name=np.asarray(args.clip_version),
        clip_download_root=np.asarray(args.clip_download_root),
        dim=np.asarray(512, dtype=np.int32),
    )
    (cache_root / "text_embeddings_index.json").write_text(
        json.dumps(
            {
                "clip_model_name": args.clip_version,
                "clip_download_root": args.clip_download_root,
                "dim": 512,
                "encoding": "clip_model.encode_text (pooled / EOT projection)",
                "n_unique_texts": len(unique_texts),
                "n_total_manifest_records": len(texts),
                "index": index,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[clip] wrote text_embeddings_clip_vit_b32.npz  shape=({len(unique_texts)}, 512)")
    print(f"[clip] wrote text_embeddings_index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
