"""Python launcher for the Round-29 ablation matrix.

Drives the per-variant train + diagnostic + summarize pipeline from the
manifest at ``analyses/round29_stage2_cond_ablation_manifest.json``.

Per the Codex post-review prompt (2026-05-26), this launcher:

  * Accepts both prompt-style aliases (`--group injection`, `--group content`,
    `--group coarse`, ..., `--group body`, `--group final`) AND the manifest's
    full group names (`A_injection`, `B_coarse`, ...).
  * Runs path preflight: config / init checkpoint / selection JSON /
    stage1_coarse_cache_root / dataset roots. Missing entries fail clearly
    unless `--allow-missing-diag-inputs` is set (and even then only
    diagnostics are skipped — training preflight always errors hard).
  * Calls the three diagnostics with the actual CLI they require
    (`--config --ckpt --selection-json --output-dir --bucket`). No
    `--samples-dir` is used (that argparse name does not exist).
  * Reads the diagnostic CKPT path from `<output_dir>/<diag-ckpt-name>`
    (default `final.pt`, override via `--diag-ckpt-name` or
    ROUND29_DIAG_CKPT_NAME env).
  * Resolves the bucket from the selection JSON (reads its `bucket` field,
    defaults to `train` since the canonical R29 subsets are train-bucket
    files).
  * Runs the summarizer at the end and packs results.

Usage:
    python scripts/stage_b_generator/run_round29_stage2_cond_ablation.py --group injection --dry-run
    python scripts/stage_b_generator/run_round29_stage2_cond_ablation.py --group content
    python scripts/stage_b_generator/run_round29_stage2_cond_ablation.py --only r29_a0_input_add
    python scripts/stage_b_generator/run_round29_stage2_cond_ablation.py --group A_injection --skip-eval
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "analyses" / "round29_stage2_cond_ablation_manifest.json"
LOG_DIR = ROOT / "runs" / "round29_stage2_cond_ablation"
SUMMARY_JSON = ROOT / "analyses" / "round29_stage2_cond_ablation_summary.json"
SUMMARY_MD = ROOT / "analyses" / "round29_stage2_cond_ablation_summary.md"

DIAG_SCRIPTS: dict[str, str] = {
    "sustained_contact": "scripts/stage_b_generator/round26_sustained_contact_diag.py",
    "gait":              "scripts/stage_b_generator/round26_gait_diag.py",
    "body_action":       "scripts/stage_b_generator/round28_body_action_diag.py",
}

# Codex P2: prompt-required group aliases must work, AND manifest group names
# stay valid. A few aliases map to multiple manifest groups (`content`,
# `all`); they are expanded at resolution time.
GROUP_ALIASES: dict[str, list[str]] = {
    "injection":   ["A_injection"],
    "coarse":      ["B_coarse"],
    "interaction": ["C_interaction"],
    "support":     ["D_support"],
    "body":        ["E_body"],
    "final":       ["F_final"],
    "content":     ["B_coarse", "C_interaction", "D_support", "E_body"],
    "all":         ["all"],
}


# ---------------------------------------------------------------------------
# Manifest + variant resolution
# ---------------------------------------------------------------------------

def _ensure_manifest() -> dict:
    if not MANIFEST.exists():
        print(f"[R29] Manifest missing at {MANIFEST}; running config generator...")
        ret = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/stage_b_generator/round29_make_stage2_cond_ablation_configs.py"),
            ],
            cwd=str(ROOT),
        )
        if ret.returncode != 0:
            raise RuntimeError("config generator failed")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _resolve_groups(group: str) -> list[str]:
    """Expand a `--group` value (alias or manifest name) into the set of
    manifest groups to include. Returns ['all'] for the special 'all' value.

    Unknown values raise ValueError.
    """
    if group in GROUP_ALIASES:
        return GROUP_ALIASES[group]
    # Allow exact manifest group names (A_injection, B_coarse, ...).
    if group in (
        "A_injection", "B_coarse", "C_interaction",
        "D_support", "E_body", "F_final",
    ):
        return [group]
    raise ValueError(
        f"unknown --group value {group!r}. Valid: "
        f"{sorted(GROUP_ALIASES)} or manifest names "
        "{A_injection, B_coarse, C_interaction, D_support, E_body, F_final}"
    )


def _pick_variants(manifest: dict, group: str, only: str) -> list[dict]:
    if only:
        want = set(only.split(","))
        return [v for v in manifest["variants"] if v["variant_id"] in want]
    selected_groups = _resolve_groups(group)
    if "all" in selected_groups:
        return list(manifest["variants"])
    return [v for v in manifest["variants"] if v["group"] in selected_groups]


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _selection_bucket(selection_json_path: Path, fallback: str = "train") -> str:
    """Read bucket from the selection JSON (top-level "bucket" key) or
    fallback. The canonical R29 subsets are train-bucket files, so the
    fallback is "train"."""
    if not selection_json_path.exists():
        return fallback
    try:
        data = json.loads(selection_json_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    val = data.get("bucket")
    if isinstance(val, str) and val in ("train", "val"):
        return val
    return fallback


def _preflight_variant(
    v: dict, *,
    diag_ckpt_name: str,
    allow_missing_diag_inputs: bool,
    skip_train: bool,
    skip_eval: bool,
) -> tuple[bool, list[str]]:
    """Return (ok, problems). Problems are human-readable strings.

    Training preflight (config, init_checkpoint, dataset roots) is HARD —
    failing here means we should not attempt training. Diagnostic preflight
    (ckpt, selection_json) is soft when `allow_missing_diag_inputs=True`.
    """
    import yaml
    problems: list[str] = []
    config_path = ROOT / v["config_path"]
    if not config_path.exists():
        problems.append(f"config missing: {config_path}")

    if not skip_train:
        init_ckpt = v.get("init_checkpoint", "")
        init_path = ROOT / init_ckpt if init_ckpt else None
        if not init_ckpt:
            problems.append("init_checkpoint missing in manifest row")
        elif init_path is not None and not init_path.exists():
            problems.append(
                f"init_checkpoint not on disk: {init_path} — "
                "override via --init-checkpoint or ROUND29_INIT_CKPT, or "
                "re-run the generator with the right --init-checkpoint."
            )
        # Dataset roots — parse the config's data.datasets list and check
        # each root exists. Skipping this check would let training crash
        # at metadata-load time after burning preflight + smoke. Reuse
        # one parsed YAML across the loop body to keep this cheap.
        if config_path.exists():
            try:
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                for ds in cfg.get("data", {}).get("datasets", []) or []:
                    root = ds.get("root", "")
                    if root and not Path(root).exists():
                        problems.append(
                            f"dataset root not on disk: {root} (subset={ds.get('name')}) — "
                            "re-run the generator with --data-root <correct path> or "
                            "export DATASETS_ROOT=<...>."
                        )
            except Exception as exc:  # noqa: BLE001
                problems.append(f"could not parse config to check dataset roots: {exc}")

    if not skip_eval:
        sel_file = v.get("subset_file", "")
        if not sel_file:
            problems.append("subset_file missing in manifest row")
        else:
            sel_path = ROOT / sel_file
            if not sel_path.exists():
                problems.append(f"selection JSON not on disk: {sel_path}")
            else:
                # Diag needs `selected`/`candidates`/`clips` to be non-empty.
                try:
                    data = json.loads(sel_path.read_text("utf-8"))
                    sel_list = (
                        data.get("selected")
                        or data.get("candidates")
                        or data.get("clips")
                        or []
                    )
                    if not sel_list:
                        problems.append(
                            f"selection JSON has no usable {{subset, seq_id}} "
                            f"list: {sel_path} (expected `selected`, "
                            f"`candidates`, or `clips`)."
                        )
                except Exception as exc:  # noqa: BLE001
                    problems.append(
                        f"could not parse selection JSON {sel_path}: {exc}"
                    )
        # The diagnostic checkpoint sits inside the training output_dir.
        diag_ckpt = ROOT / v["output_dir"] / diag_ckpt_name
        if not diag_ckpt.exists():
            msg = (
                f"diag ckpt not on disk: {diag_ckpt} (this is normal BEFORE "
                f"training; will be created at runs/training/<vid>/{diag_ckpt_name})"
            )
            if not skip_train:
                # Will be produced by THIS run's training, so just informational.
                pass
            else:
                problems.append(msg)

    # Hard problems are anything except the "diag ckpt missing pre-train" note.
    hard = [p for p in problems if "(this is normal BEFORE training" not in p]
    if hard and not allow_missing_diag_inputs and skip_train:
        return False, problems
    if any("config missing" in p or "init_checkpoint" in p for p in hard):
        if not allow_missing_diag_inputs:
            return False, problems
    return (len(hard) == 0 or allow_missing_diag_inputs), problems


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _train_command(
    config: str, *, single_gpu: bool, num_processes: int,
) -> list[str]:
    if single_gpu or num_processes <= 1:
        return [sys.executable, "-u", "src/piano/training/train_anchordiff.py",
                "--config", config]
    return [
        "accelerate", "launch",
        "--num_processes", str(num_processes),
        "--multi_gpu",
        "--mixed_precision", "bf16",
        "src/piano/training/train_anchordiff.py",
        "--config", config,
    ]


def _diag_commands(
    v: dict, *, diag_ckpt_name: str,
) -> list[tuple[str, list[str], Path]]:
    """Build the three diag commands using the REAL diagnostic CLI:
    --config --ckpt --selection-json --output-dir --bucket.

    Returns: list of (kind, cmd, output_dir).
    """
    config_path = v["config_path"]   # repo-relative
    output_dir = v["output_dir"]
    # Single subset file shared by trainer + diag — the diag scripts
    # now read `clips` (which the train_indices builder emits) in
    # addition to `selected`/`candidates`.
    subset_file = v["subset_file"]
    bucket = _selection_bucket(ROOT / subset_file)
    ckpt_path = f"{output_dir}/{diag_ckpt_name}"

    cmds: list[tuple[str, list[str], Path]] = []
    for kind, script in DIAG_SCRIPTS.items():
        out_dir = ROOT / "analyses" / f"round29_{v['variant_id']}_diag_{kind}"
        cmd = [
            sys.executable, "-u", script,
            "--config", config_path,
            "--ckpt", ckpt_path,
            "--selection-json", subset_file,
            "--output-dir", str(out_dir.relative_to(ROOT).as_posix()),
            "--bucket", bucket,
        ]
        cmds.append((kind, cmd, out_dir))
    return cmds


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[R29] $ {' '.join(shlex.quote(s) for s in cmd)}")
    with log_file.open("ab") as f:
        ret = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
    return ret.returncode


def _print_dry_run(label: str, cmd: list[str]) -> None:
    print(f"[R29 DRY-RUN {label}]")
    print(f"    $ {' '.join(shlex.quote(s) for s in cmd)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Python launcher for Round-29 Stage-2 cond + injection ablation."
    )
    parser.add_argument(
        "--group", default="all",
        help=(
            "Group selector. Aliases: injection / coarse / interaction / "
            "support / body / final / content / all. Manifest names also "
            "accepted (A_injection, B_coarse, ...). Ignored if --only is set."
        ),
    )
    parser.add_argument("--only", default="",
                        help="Comma-separated variant ids to run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print exactly what would run; touch no disk.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Bypass path preflight (use with care).")
    parser.add_argument("--allow-missing-diag-inputs", action="store_true",
                        help="Continue past missing diag ckpt / selection JSON.")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Run training on one GPU (no accelerate launch). "
                             "Equivalent to --num-processes 1.")
    parser.add_argument(
        "--num-processes", type=int,
        default=int(os.environ.get("ROUND29_NUM_PROCESSES", "0")),
        help=(
            "Number of accelerate processes (= number of GPUs to use). "
            "Default: ROUND29_NUM_PROCESSES env var or auto-detect via "
            "CUDA_VISIBLE_DEVICES / torch.cuda.device_count() (capped). "
            "Pass 1 for single-GPU mode (equivalent to --single-gpu)."
        ),
    )
    parser.add_argument(
        "--diag-ckpt-name",
        default=os.environ.get("ROUND29_DIAG_CKPT_NAME", "final.pt"),
        help="Filename of the checkpoint diagnostics should evaluate "
             "(under <output_dir>/). Default: final.pt.",
    )
    args = parser.parse_args()

    # Auto-detect num_processes if not set explicitly.
    if args.num_processes <= 0:
        try:
            import torch  # noqa: PLC0415
            args.num_processes = max(1, torch.cuda.device_count())
        except Exception:
            args.num_processes = 1
        print(f"[R29] auto-detected num_processes = {args.num_processes} "
              f"(override via --num-processes or ROUND29_NUM_PROCESSES env)")

    manifest = _ensure_manifest()
    variants = _pick_variants(manifest, args.group, args.only)
    if not variants:
        print(f"[R29] no variants matched group='{args.group}' only='{args.only}'")
        return 0
    print(f"[R29] {len(variants)} variant(s) to process "
          f"(group='{args.group}', only='{args.only}').")

    # Preflight up front so the user sees all failures together rather
    # than after training partially burns GPU time.
    if not args.skip_preflight and not args.dry_run:
        any_hard = False
        print("[R29] Running preflight on all selected variants...")
        for v in variants:
            ok, problems = _preflight_variant(
                v,
                diag_ckpt_name=args.diag_ckpt_name,
                allow_missing_diag_inputs=args.allow_missing_diag_inputs,
                skip_train=args.skip_train,
                skip_eval=args.skip_eval,
            )
            for p in problems:
                print(f"    [{v['variant_id']}] {p}")
            if not ok:
                any_hard = True
        if any_hard:
            print(
                "[R29] FATAL: hard preflight failures above. "
                "Fix them, or pass --skip-preflight to bypass."
            )
            return 1

    # Smoke test (fast dry-run mode).
    if not args.dry_run:
        print("[R29] Smoke test (fast)...")
        ret = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/stage_b_generator/round29_stage2_cond_smoke_test.py"),
                "--dry-run",
            ],
            cwd=str(ROOT),
        )
        if ret.returncode != 0:
            print("[R29] FATAL: smoke test failed — fix before launching trains.")
            return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for v in variants:
        vid = v["variant_id"]
        log_path = LOG_DIR / f"{vid}.log"
        t0 = time.time()
        print()
        print("================================================================")
        print(f"[{datetime.now():%F %T}] BEGIN {vid}  group={v['group']}")
        print(f"    config: {v['config_path']}")
        print(f"    output: {v['output_dir']}")
        print(f"    log:    {log_path}")
        print("================================================================")

        # TRAIN
        train_cmd = _train_command(
            v["config_path"],
            single_gpu=args.single_gpu,
            num_processes=args.num_processes,
        )
        if args.skip_train:
            print(f"--skip-train: skipping training for {vid}")
        elif args.dry_run:
            _print_dry_run(f"{vid} TRAIN", train_cmd)
        else:
            ret = _run(train_cmd, log_path)
            if ret != 0:
                print(f"[R29] WARN: training failed for {vid} (rc={ret}); skipping diag")
                continue

        # DIAGNOSTICS
        if args.skip_eval:
            print(f"--skip-eval: skipping diag for {vid}")
        else:
            for kind, cmd, out_dir in _diag_commands(
                v, diag_ckpt_name=args.diag_ckpt_name,
            ):
                if args.dry_run:
                    _print_dry_run(f"{vid} DIAG/{kind}", cmd)
                    continue
                # Re-check the ckpt now (in real mode after training).
                ckpt = ROOT / v["output_dir"] / args.diag_ckpt_name
                if not ckpt.exists() and not args.allow_missing_diag_inputs:
                    print(
                        f"[R29] FATAL: diag ckpt missing for {vid}/{kind}: "
                        f"{ckpt} — pass --allow-missing-diag-inputs to skip."
                    )
                    return 2
                out_dir.mkdir(parents=True, exist_ok=True)
                _run(cmd, log_path)

        print(f"[{datetime.now():%F %T}] DONE  {vid} in {time.time() - t0:.0f}s")

    # SUMMARIZE
    if not args.dry_run:
        print()
        print("================================================================")
        print(f"[{datetime.now():%F %T}] Summarizing...")
        subprocess.run(
            [
                sys.executable, "-u",
                str(ROOT / "scripts/stage_b_generator/round29_summarize_stage2_cond_ablation.py"),
                "--manifest", str(MANIFEST),
                "--output-json", str(SUMMARY_JSON),
                "--output-md", str(SUMMARY_MD),
                "--allow-missing-results",
            ],
            cwd=str(ROOT),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
