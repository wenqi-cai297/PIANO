"""Round-36 dynamics-loss config generator.

Generates one Stage-1 and one Stage-1.5 training config for the temporal
dynamics follow-up:

  - Stage-1: R31 V8.6 substrate + raw velocity/acceleration losses.
  - Stage-1.5: R34 V2-A substrate + raw C41 velocity/acceleration losses.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT / "configs" / "training"
DEFAULT_ANALYSES_DIR = ROOT / "analyses"
DEFAULT_DATA_ROOT = "E:/Project/Datasets/InterAct/piano_official_process_4"

STAGE1_BASE_NAME = "stage1_v8_v6_full_f1"
STAGE1_OUT_NAME = "stage1_r36_v8v6_dynacc"
STAGE1P5_BASE_NAME = "stage1p5_r34v2_a_lambda0p005"
STAGE1P5_OUT_NAME = "stage1p5_r36_r34v2_a_c41dyn"


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def _ensure_base_configs(
    *,
    data_root: str,
    config_dir: Path,
    analyses_dir: Path,
    dry_run: bool,
    regen: bool,
) -> None:
    stage1_base = config_dir / f"{STAGE1_BASE_NAME}.yaml"
    stage1p5_r33_base = config_dir / "stage1p5_r33_v1_xattn.yaml"
    stage1p5_base = config_dir / f"{STAGE1P5_BASE_NAME}.yaml"

    if regen or not stage1_base.exists():
        _run(
            [
                sys.executable,
                "scripts/stage_a_generator/round31_make_stage1_v8_configs.py",
                "--data-root",
                data_root,
                "--config-dir",
                str(config_dir),
                "--analyses-dir",
                str(analyses_dir),
            ],
            dry_run=dry_run,
        )

    if regen or not stage1p5_r33_base.exists():
        _run(
            [
                sys.executable,
                "scripts/stage_a_generator/round33_make_stage1p5_v0_configs.py",
                "--data-root",
                data_root,
                "--config-dir",
                str(config_dir),
                "--analyses-dir",
                str(analyses_dir),
            ],
            dry_run=dry_run,
        )

    if regen or not stage1p5_base.exists():
        _run(
            [
                sys.executable,
                "scripts/stage_a_generator/round34v2_make_stage1p5_configs.py",
                "--base-cfg",
                str(stage1p5_r33_base),
                "--out-dir",
                str(config_dir),
            ],
            dry_run=dry_run,
        )


def _write_stage1_config(config_dir: Path, *, dry_run: bool) -> dict:
    base_path = config_dir / f"{STAGE1_BASE_NAME}.yaml"
    out_path = config_dir / f"{STAGE1_OUT_NAME}.yaml"
    if not dry_run and not base_path.exists():
        raise SystemExit(f"missing Stage-1 base config: {base_path}")

    if dry_run:
        print(f"DRY-RUN would write {out_path} from {base_path}")
    else:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(base_path)
        cfg.output_dir = f"runs/training/{STAGE1_OUT_NAME}"
        cfg.logging.run_name = STAGE1_OUT_NAME
        cfg.loss.w_r36_raw_velocity = 0.05
        cfg.loss.w_r36_raw_acceleration = 0.10
        cfg.loss.r36_raw_dynamics_channel_subset = []
        cfg.loss.r36_raw_dynamics_normalize_by_gt_std = True
        OmegaConf.save(cfg, out_path)
        print(f"wrote {out_path}")

    return {
        "variant_id": STAGE1_OUT_NAME,
        "stage": "stage1",
        "base_config": str(base_path),
        "config_path": str(out_path),
        "output_dir": f"runs/training/{STAGE1_OUT_NAME}",
        "loss": {
            "w_r36_raw_velocity": 0.05,
            "w_r36_raw_acceleration": 0.10,
            "r36_raw_dynamics_channel_subset": [],
            "r36_raw_dynamics_normalize_by_gt_std": True,
        },
        "notes": (
            "R31 V8.6 substrate; adds raw-space velocity and acceleration "
            "matching on all 23 Stage-1 channels."
        ),
    }


def _write_stage1p5_config(config_dir: Path, *, dry_run: bool) -> dict:
    base_path = config_dir / f"{STAGE1P5_BASE_NAME}.yaml"
    out_path = config_dir / f"{STAGE1P5_OUT_NAME}.yaml"
    if not dry_run and not base_path.exists():
        raise SystemExit(f"missing Stage-1.5 base config: {base_path}")

    if dry_run:
        print(f"DRY-RUN would write {out_path} from {base_path}")
    else:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(base_path)
        cfg.output_dir = f"runs/training/{STAGE1P5_OUT_NAME}"
        cfg.logging.run_name = STAGE1P5_OUT_NAME
        cfg.loss.w_r36_c41_velocity = 0.05
        cfg.loss.w_r36_c41_acceleration = 0.10
        cfg.loss.r36_c41_dynamics_channel_subset = []
        cfg.loss.r36_c41_dynamics_normalize_by_gt_std = True
        OmegaConf.save(cfg, out_path)
        print(f"wrote {out_path}")

    return {
        "variant_id": STAGE1P5_OUT_NAME,
        "stage": "stage1p5",
        "base_config": str(base_path),
        "config_path": str(out_path),
        "output_dir": f"runs/training/{STAGE1P5_OUT_NAME}",
        "loss": {
            "w_r36_c41_velocity": 0.05,
            "w_r36_c41_acceleration": 0.10,
            "r36_c41_dynamics_channel_subset": [],
            "r36_c41_dynamics_normalize_by_gt_std": True,
        },
        "notes": (
            "R34 V2-A substrate; keeps wrist low-band lambda=0.005 and "
            "adds raw C41 velocity/acceleration matching."
        ),
    }


def _write_manifest(analyses_dir: Path, rows: list[dict], *, dry_run: bool) -> None:
    manifest_json = analyses_dir / "round36_dynamics_manifest.json"
    manifest_md = analyses_dir / "round36_dynamics_manifest.md"
    if dry_run:
        print(f"DRY-RUN would write {manifest_json}")
        print(f"DRY-RUN would write {manifest_md}")
        return

    analyses_dir.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(
        json.dumps({"variants": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Round-36 Dynamics-Loss Manifest",
        "",
        "Temporal dynamics follow-up for Stage-1 and Stage-1.5.",
        "",
        "| variant | stage | base | velocity_w | acceleration_w |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        loss = row["loss"]
        vel_w = loss.get("w_r36_raw_velocity", loss.get("w_r36_c41_velocity"))
        acc_w = loss.get("w_r36_raw_acceleration", loss.get("w_r36_c41_acceleration"))
        lines.append(
            f"| `{row['variant_id']}` | {row['stage']} | "
            f"`{Path(row['base_config']).name}` | {vel_w} | {acc_w} |"
        )
    manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {manifest_json}")
    print(f"wrote {manifest_md}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--regen-bases", action="store_true")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATASETS_ROOT", DEFAULT_DATA_ROOT),
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--analyses-dir", type=Path, default=DEFAULT_ANALYSES_DIR)
    args = parser.parse_args()

    config_dir = args.config_dir
    analyses_dir = args.analyses_dir
    _ensure_base_configs(
        data_root=args.data_root,
        config_dir=config_dir,
        analyses_dir=analyses_dir,
        dry_run=bool(args.dry_run),
        regen=bool(args.regen_bases) or ("DATASETS_ROOT" in os.environ),
    )
    rows = [
        _write_stage1_config(config_dir, dry_run=bool(args.dry_run)),
        _write_stage1p5_config(config_dir, dry_run=bool(args.dry_run)),
    ]
    _write_manifest(analyses_dir, rows, dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
