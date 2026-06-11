#!/usr/bin/env bash
# Round-42 condition-consistency 2x2 diagnostic.
#
# Goal
# ----
# Separate "bad Stage-1 coarse" from "Stage-1 / Stage-1.5 cond mismatch".
# Runs the same frozen PB1 downstream diagnostics under four condition
# combinations:
#
#   OO: oracle   stage1_coarse + oracle   C41/S4
#   GO: generated stage1_coarse + oracle   C41/S4
#   OG: oracle   stage1_coarse + generated C41/S4 from Stage-1.5(oracle S1)
#   GG: generated stage1_coarse + generated C41/S4 from Stage-1.5(generated S1)
#
# The key readout:
#   - GO bad but GG better  -> oracle C41/S4 mismatches generated Stage-1.
#   - OG good but GG bad    -> Stage-1.5 is OOD when fed generated Stage-1.
#   - OO good, all others bad -> both upstream predictors need cascade training.
#
# Defaults are intentionally "latest ship references" used by R41/R40:
#   Stage-1  : V8 V6
#   Stage-1.5: R38-B1 init-pose
#   PB1      : R29 PB1 AdaLN-S4
#
# Usage:
#   export DATASETS_ROOT=/media/8TB_data/Cai/datasets/InterAct/piano_official_process_4
#   bash scripts/stage_a_generator/run_round42_cond_2x2_diag.sh
#
# Common overrides:
#   ROUND42_2X2_BUCKETS="val"                 # default val
#   ROUND42_2X2_CELLS="oo go og gg"           # default all four
#   ROUND42_2X2_STAGE1_CFG=...
#   ROUND42_2X2_STAGE1_CKPT=...
#   ROUND42_2X2_STAGE1P5_CFG=...
#   ROUND42_2X2_STAGE1P5_CKPT=...
#   ROUND42_2X2_PB1_CKPT=...
#   ROUND42_2X2_PACK_NPZ=1                    # default 1; set 0 to omit caches

set -euo pipefail
cd "$(dirname "$0")/../.."

ONLY_CELLS="${ROUND42_2X2_CELLS:-oo go og gg}"
DRY_RUN=0
SKIP_SAMPLE_STAGE1=0
SKIP_SAMPLE_STAGE1P5=0
SKIP_DIAG=0
FORCE=0
ALLOW_PARTIAL="${ROUND42_2X2_ALLOW_PARTIAL:-0}"
PACK_NPZ="${ROUND42_2X2_PACK_NPZ:-1}"

SEED="${ROUND42_2X2_SEED:-42}"
CFG_SCALE="${ROUND42_2X2_CFG_SCALE:-1.0}"
SAMPLER="${ROUND42_2X2_SAMPLER:-ddim_eta0}"
BUCKETS_STR="${ROUND42_2X2_BUCKETS:-val}"

STAGE1_CFG="${ROUND42_2X2_STAGE1_CFG:-configs/training/stage1_v8_v6_full_f1.yaml}"
STAGE1_CKPT="${ROUND42_2X2_STAGE1_CKPT:-runs/training/stage1_v8_v6_full_f1/final.pt}"
STAGE1P5_CFG="${ROUND42_2X2_STAGE1P5_CFG:-configs/training/stage1p5_r38_b1_init_pose.yaml}"
STAGE1P5_CKPT="${ROUND42_2X2_STAGE1P5_CKPT:-runs/training/stage1p5_r38_b1_init_pose/final.pt}"
PB1_VARIANT="r29_pb_a1_adaln_s4"
PB1_CFG="${ROUND42_2X2_PB1_CFG:-configs/training/anchordiff_${PB1_VARIANT}.yaml}"
PB1_CKPT="${ROUND42_2X2_PB1_CKPT:-runs/training/stageB_anchordiff_${PB1_VARIANT}/final.pt}"

SELECTION_TRAIN="${ROUND42_2X2_SELECTION_TRAIN:-analyses/round27_tier0_train_indices_48_balanced.json}"
SELECTION_VAL="${ROUND42_2X2_SELECTION_VAL:-analyses/round29_val_diag_indices_48_balanced.json}"

STAMP="${ROUND42_2X2_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${ROUND42_2X2_OUT_ROOT:-analyses/round42_cond_2x2_${STAMP}}"
LOG_DIR="${ROUND42_2X2_LOG_DIR:-runs/round42_cond_2x2_${STAMP}}"
TARBALL="${ROUND42_2X2_TARBALL:-${OUT_ROOT}.tar.gz}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cells)                ONLY_CELLS="$2"; shift 2 ;;
        --buckets)              BUCKETS_STR="$2"; shift 2 ;;
        --dry-run)              DRY_RUN=1; shift ;;
        --skip-sample-stage1)   SKIP_SAMPLE_STAGE1=1; shift ;;
        --skip-sample-stage1p5) SKIP_SAMPLE_STAGE1P5=1; shift ;;
        --skip-diag)            SKIP_DIAG=1; shift ;;
        --force)                FORCE=1; shift ;;
        --no-pack-npz)          PACK_NPZ=0; shift ;;
        -h|--help)              sed -n '1,72p' "$0"; exit 0 ;;
        *) echo "[R42 2x2] Unknown arg: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/summary_${STAMP}.log"

if [[ -z "${PY:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PY="python";
    elif command -v python3 >/dev/null 2>&1; then PY="python3";
    else echo "[R42 2x2] FATAL: neither python nor python3 was found" >&2; exit 127; fi
fi

log() { echo "$@" | tee -a "${SUMMARY_LOG}"; }

normalize_list() {
    echo "$1" | tr ',' ' '
}

CELLS_STR="$(normalize_list "${ONLY_CELLS}")"
BUCKETS_STR="$(normalize_list "${BUCKETS_STR}")"
# shellcheck disable=SC2206
CELLS=(${CELLS_STR})
# shellcheck disable=SC2206
BUCKETS=(${BUCKETS_STR})

cell_label() {
    case "$1" in
        oo) echo "oo_oracle_s1_oracle_s1p5" ;;
        go) echo "go_generated_s1_oracle_s1p5" ;;
        og) echo "og_oracle_s1_generated_s1p5" ;;
        gg) echo "gg_generated_s1_generated_s1p5" ;;
        *) echo "[R42 2x2] unknown cell '$1'" >&2; return 1 ;;
    esac
}

need_stage1_sample=0
need_stage1p5_oracle=0
need_stage1p5_generated=0
for c in "${CELLS[@]}"; do
    case "${c}" in
        oo) ;;
        go) need_stage1_sample=1 ;;
        og) need_stage1p5_oracle=1 ;;
        gg) need_stage1_sample=1; need_stage1p5_generated=1 ;;
        *) echo "[R42 2x2] FATAL: unknown cell '${c}' (valid: oo go og gg)" >&2; exit 2 ;;
    esac
done

select_json_for_bucket() {
    case "$1" in
        train) echo "${SELECTION_TRAIN}" ;;
        val)   echo "${SELECTION_VAL}" ;;
        *) echo "[R42 2x2] unknown bucket '$1'" >&2; return 1 ;;
    esac
}

S1_ROOT="${OUT_ROOT}/substitute_conds/stage1_generated"
S1P5_ORACLE_ROOT="${OUT_ROOT}/substitute_conds/stage1p5_from_oracle_s1"
S1P5_GEN_RAW_ROOT="${OUT_ROOT}/substitute_conds/stage1p5_from_generated_s1_raw"
GG_MERGED_ROOT="${OUT_ROOT}/substitute_conds/merged_generated_s1_generated_s1p5"
DIAG_ROOT="${OUT_ROOT}/diag"

log "===== R42 condition 2x2 diagnostic ${STAMP} ====="
log "cells: ${CELLS[*]}"
log "buckets: ${BUCKETS[*]}"
log "stage1:    ${STAGE1_CFG} | ${STAGE1_CKPT}"
log "stage1.5:  ${STAGE1P5_CFG} | ${STAGE1P5_CKPT}"
log "PB1:       ${PB1_CFG} | ${PB1_CKPT}"
log "sampler=${SAMPLER} cfg_scale=${CFG_SCALE} seed=${SEED}"
log "out_root=${OUT_ROOT}"
log

preflight_fail=0
if [[ ${DRY_RUN} -eq 0 ]]; then
    for p in "${STAGE1_CFG}" "${STAGE1P5_CFG}" "${PB1_CFG}"; do
        [[ ! -f "${p}" ]] && { log "[R42 PREFLIGHT FAIL] missing config: ${p}"; preflight_fail=1; }
    done
    for p in "${PB1_CKPT}"; do
        [[ ! -f "${p}" ]] && { log "[R42 PREFLIGHT FAIL] missing ckpt: ${p}"; preflight_fail=1; }
    done
    if [[ ${need_stage1_sample} -eq 1 ]]; then
        [[ ! -f "${STAGE1_CKPT}" ]] && { log "[R42 PREFLIGHT FAIL] missing Stage-1 ckpt: ${STAGE1_CKPT}"; preflight_fail=1; }
    fi
    if [[ ${need_stage1p5_oracle} -eq 1 || ${need_stage1p5_generated} -eq 1 ]]; then
        [[ ! -f "${STAGE1P5_CKPT}" ]] && { log "[R42 PREFLIGHT FAIL] missing Stage-1.5 ckpt: ${STAGE1P5_CKPT}"; preflight_fail=1; }
    fi
    for b in "${BUCKETS[@]}"; do
        sel="$(select_json_for_bucket "${b}")"
        [[ ! -f "${sel}" ]] && { log "[R42 PREFLIGHT FAIL] missing selection for ${b}: ${sel}"; preflight_fail=1; }
    done
fi
for s in scripts/stage_a_generator/sample_substitute_conds_cli.py \
         scripts/stage_b_generator/round26_sustained_contact_diag.py \
         scripts/stage_b_generator/round26_gait_diag.py \
         scripts/stage_b_generator/round28_body_action_diag.py \
         scripts/stage_b_generator/round29_g1_soft_stance_diag.py; do
    [[ ! -f "${s}" ]] && { log "[R42 PREFLIGHT FAIL] missing script: ${s}"; preflight_fail=1; }
done
if [[ ${preflight_fail} -ne 0 ]]; then
    log "[R42] FATAL preflight failures."
    exit 1
fi

sample_stage1() {
    local bucket="$1"
    local sel="$2"
    local out="${S1_ROOT}/${bucket}"
    local done_marker="${out}/.round42_done"
    if [[ ${SKIP_SAMPLE_STAGE1} -eq 1 ]]; then
        log "[R42] skip Stage-1 sample for ${bucket}; using ${out}"
        [[ ${DRY_RUN} -eq 1 || -d "${out}" ]] || { log "[R42] missing Stage-1 cache: ${out}"; return 1; }
        return 0
    fi
    if [[ -f "${done_marker}" && ${FORCE} -eq 0 ]]; then
        log "[R42] Stage-1 cache already exists for ${bucket}: ${out}"
        return 0
    fi
    local log_path="${LOG_DIR}/sample_stage1_${bucket}.log"
    local cmd=("${PY}" -u scripts/stage_a_generator/sample_substitute_conds_cli.py
        --stage stage1
        --config "${STAGE1_CFG}"
        --ckpt "${STAGE1_CKPT}"
        --selection-json "${sel}"
        --bucket "${bucket}"
        --out-dir "${out}"
        --seed "${SEED}"
        --cfg-scale "${CFG_SCALE}"
        --sampler "${SAMPLER}")
    log
    log "================================================================"
    log "[$(date '+%F %T')] SAMPLE Stage-1 generated (${bucket}) -> ${out}"
    log "================================================================"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R42 DRY-RUN] ${cmd[*]}"
    else
        mkdir -p "${out}"
        "${cmd[@]}" 2>&1 | tee "${log_path}"
        touch "${done_marker}"
    fi
}

sample_stage1p5() {
    local mode="$1"  # oracle or generated
    local bucket="$2"
    local sel="$3"
    local out upstream done_marker log_path
    if [[ "${mode}" == "oracle" ]]; then
        out="${S1P5_ORACLE_ROOT}/${bucket}"
        upstream=""
        log_path="${LOG_DIR}/sample_stage1p5_oracle_s1_${bucket}.log"
    else
        out="${S1P5_GEN_RAW_ROOT}/${bucket}"
        upstream="${S1_ROOT}/${bucket}"
        log_path="${LOG_DIR}/sample_stage1p5_generated_s1_${bucket}.log"
    fi
    done_marker="${out}/.round42_done"
    if [[ ${SKIP_SAMPLE_STAGE1P5} -eq 1 ]]; then
        log "[R42] skip Stage-1.5 sample (${mode}) for ${bucket}; using ${out}"
        [[ ${DRY_RUN} -eq 1 || -d "${out}" ]] || { log "[R42] missing Stage-1.5 cache: ${out}"; return 1; }
        return 0
    fi
    if [[ -f "${done_marker}" && ${FORCE} -eq 0 ]]; then
        log "[R42] Stage-1.5 cache already exists (${mode}, ${bucket}): ${out}"
        return 0
    fi
    local cmd=("${PY}" -u scripts/stage_a_generator/sample_substitute_conds_cli.py
        --stage stage1p5
        --config "${STAGE1P5_CFG}"
        --ckpt "${STAGE1P5_CKPT}"
        --selection-json "${sel}"
        --bucket "${bucket}"
        --out-dir "${out}"
        --seed "${SEED}"
        --cfg-scale "${CFG_SCALE}"
        --sampler "${SAMPLER}")
    if [[ -n "${upstream}" ]]; then
        cmd+=(--upstream-dir "${upstream}")
    fi
    log
    log "================================================================"
    if [[ "${mode}" == "oracle" ]]; then
        log "[$(date '+%F %T')] SAMPLE Stage-1.5 with ORACLE Stage-1 (${bucket}) -> ${out}"
    else
        log "[$(date '+%F %T')] SAMPLE Stage-1.5 with GENERATED Stage-1 (${bucket}) -> ${out}"
    fi
    log "================================================================"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R42 DRY-RUN] ${cmd[*]}"
    else
        mkdir -p "${out}"
        "${cmd[@]}" 2>&1 | tee "${log_path}"
        touch "${done_marker}"
    fi
}

merge_gg_cache() {
    local bucket="$1"
    local s1="${S1_ROOT}/${bucket}"
    local s1p5="${S1P5_GEN_RAW_ROOT}/${bucket}"
    local out="${GG_MERGED_ROOT}/${bucket}"
    local done_marker="${out}/.round42_done"
    if [[ -f "${done_marker}" && ${FORCE} -eq 0 ]]; then
        log "[R42] GG merged cache already exists for ${bucket}: ${out}"
        return 0
    fi
    log
    log "================================================================"
    log "[$(date '+%F %T')] MERGE GG cache (${bucket}) -> ${out}"
    log "================================================================"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "[R42 DRY-RUN] merge ${s1} + ${s1p5} -> ${out}"
        return 0
    fi
    "${PY}" -u - "${s1}" "${s1p5}" "${out}" <<'PY'
import sys
from pathlib import Path

import numpy as np

s1 = Path(sys.argv[1])
s1p5 = Path(sys.argv[2])
out = Path(sys.argv[3])
if not s1.exists():
    raise SystemExit(f"missing Stage-1 cache: {s1}")
if not s1p5.exists():
    raise SystemExit(f"missing Stage-1.5 cache: {s1p5}")
n = 0
for p in s1p5.rglob("*.npz"):
    rel = p.relative_to(s1p5)
    s1_p = s1 / rel
    if not s1_p.exists():
        raise SystemExit(f"missing Stage-1 cache for {rel}")
    s1_data = dict(np.load(s1_p))
    s1p5_data = dict(np.load(p))
    out_p = out / rel
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_p,
        stage1_coarse=s1_data["stage1_coarse"],
        stage2_coarse_extra=s1p5_data["stage2_coarse_extra"],
        stage2_support=s1p5_data["stage2_support"],
        valid_T=s1p5_data.get("valid_T", s1_data.get("valid_T")),
        seed=s1p5_data.get("seed", s1_data.get("seed")),
    )
    n += 1
print(f"merged {n} clips")
PY
    touch "${done_marker}"
}

run_diag() {
    local cell="$1"
    local bucket="$2"
    local kind="$3"
    local sel="$4"
    local label sub_dir script out_dir log_path summary_file
    label="$(cell_label "${cell}")"
    case "${cell}" in
        oo) sub_dir="" ;;
        go) sub_dir="${S1_ROOT}/${bucket}" ;;
        og) sub_dir="${S1P5_ORACLE_ROOT}/${bucket}" ;;
        gg) sub_dir="${GG_MERGED_ROOT}/${bucket}" ;;
    esac
    case "${kind}" in
        sustained_contact)
            script="scripts/stage_b_generator/round26_sustained_contact_diag.py"
            summary_file="sustained_contact_summary.md"
            ;;
        gait)
            script="scripts/stage_b_generator/round26_gait_diag.py"
            summary_file="gait_summary.md"
            ;;
        body_action)
            script="scripts/stage_b_generator/round28_body_action_diag.py"
            summary_file="body_action_summary.md"
            ;;
        g1_soft_stance)
            script="scripts/stage_b_generator/round29_g1_soft_stance_diag.py"
            summary_file="g1_soft_stance_summary.md"
            ;;
        *) echo "[R42 2x2] unknown diag kind ${kind}" >&2; return 1 ;;
    esac
    out_dir="${DIAG_ROOT}/${label}/${kind}_${bucket}"
    log_path="${LOG_DIR}/diag_${label}_${kind}_${bucket}.log"
    if [[ -f "${out_dir}/${summary_file}" && ${FORCE} -eq 0 ]]; then
        log "[R42] diag exists; skip ${label}/${kind}_${bucket}"
        return 0
    fi
    local cmd=("${PY}" -u "${script}"
        --config "${PB1_CFG}"
        --ckpt "${PB1_CKPT}"
        --selection-json "${sel}"
        --output-dir "${out_dir}"
        --bucket "${bucket}"
        --cfg-scale "${CFG_SCALE}"
        --seed "${SEED}")
    if [[ -n "${sub_dir}" ]]; then
        cmd+=(--substitute-conds-dir "${sub_dir}")
    fi
    log "[R42 DIAG] ${label}/${kind}_${bucket}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log "    ${cmd[*]}"
        return 0
    fi
    mkdir -p "${out_dir}"
    if ! "${cmd[@]}" 2>&1 | tee "${log_path}"; then
        if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
            log "[R42] WARN: diag failed but continuing: ${label}/${kind}_${bucket}"
            return 0
        fi
        log "[R42] FATAL: diag failed: ${label}/${kind}_${bucket}"
        return 1
    fi
}

for b in "${BUCKETS[@]}"; do
    sel="$(select_json_for_bucket "${b}")"
    if [[ ${need_stage1_sample} -eq 1 ]]; then
        sample_stage1 "${b}" "${sel}"
    fi
    if [[ ${need_stage1p5_oracle} -eq 1 ]]; then
        sample_stage1p5 oracle "${b}" "${sel}"
    fi
    if [[ ${need_stage1p5_generated} -eq 1 ]]; then
        sample_stage1p5 generated "${b}" "${sel}"
        merge_gg_cache "${b}"
    fi
done

if [[ ${SKIP_DIAG} -eq 0 ]]; then
    log
    log "================================================================"
    log "[$(date '+%F %T')] RUN PB1 DIAGS"
    log "================================================================"
    for c in "${CELLS[@]}"; do
        for b in "${BUCKETS[@]}"; do
            sel="$(select_json_for_bucket "${b}")"
            for kind in sustained_contact gait body_action g1_soft_stance; do
                run_diag "${c}" "${b}" "${kind}" "${sel}"
            done
        done
    done
else
    log "[R42] --skip-diag: skipping PB1 diags"
fi

SUMMARY_MD="${OUT_ROOT}/round42_cond_2x2_summary.md"
if [[ ${DRY_RUN} -eq 0 ]]; then
    "${PY}" -u - "${OUT_ROOT}" "${SUMMARY_MD}" "${CELLS_STR}" "${BUCKETS_STR}" <<'PY'
import re
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_md = Path(sys.argv[2])
cells = sys.argv[3].split()
buckets = sys.argv[4].split()

labels = {
    "oo": "OO oracle S1 + oracle C41/S4",
    "go": "GO generated S1 + oracle C41/S4",
    "og": "OG oracle S1 + generated C41/S4",
    "gg": "GG generated S1 + generated C41/S4",
}
dirs = {
    "oo": "oo_oracle_s1_oracle_s1p5",
    "go": "go_generated_s1_oracle_s1p5",
    "og": "og_oracle_s1_generated_s1p5",
    "gg": "gg_generated_s1_generated_s1p5",
}

def read(path: Path) -> str:
    return path.read_text("utf-8", errors="replace") if path.exists() else ""

def mfloat(text: str, pat: str):
    m = re.search(pat, text, re.MULTILINE)
    return float(m.group(1)) if m else None

def fmt(v, nd=2):
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"

def part_metric(text: str, part: str, col: int):
    # | left_hand | 60 | 30.87 | 66.04 | +10.01 | 0.992 | 4.3% |
    for line in text.splitlines():
        if line.startswith(f"| {part} |"):
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            try:
                return float(cols[col])
            except Exception:
                return None
    return None

def body_metric(text: str, joint: str, col: int):
    for line in text.splitlines():
        if line.startswith(f"| {joint} |"):
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            try:
                return float(cols[col])
            except Exception:
                return None
    return None

rows = []
for bucket in buckets:
    for cell in cells:
        root = out_root / "diag" / dirs[cell]
        sust = read(root / f"sustained_contact_{bucket}" / "sustained_contact_summary.md")
        gait = read(root / f"gait_{bucket}" / "gait_summary.md")
        body = read(root / f"body_action_{bucket}" / "body_action_summary.md")
        g1 = read(root / f"g1_soft_stance_{bucket}" / "g1_soft_stance_summary.md")
        rows.append({
            "bucket": bucket,
            "cell": cell,
            "label": labels[cell],
            "sust_drift": mfloat(sust, r"drift_max_cm:\s+mean=([\d.]+)"),
            "sust_p95": mfloat(sust, r"drift_max_cm:.*?p95=([\d.]+)"),
            "track": mfloat(sust, r"tracking_fraction .*?: mean=([\d.]+)"),
            "lh": part_metric(sust, "left_hand", 2),
            "rh": part_metric(sust, "right_hand", 2),
            "pelvis": part_metric(sust, "pelvis", 2),
            "gait_pred_trans": mfloat(gait, r"\| transitions/sec \| [\d.\-]+ \| ([\d.\-]+) \|"),
            "gait_gt_trans": mfloat(gait, r"\| transitions/sec \| ([\d.\-]+) \|"),
            "body_lw": body_metric(body, "left_wrist", 1),
            "body_rw": body_metric(body, "right_wrist", 1),
            "g1_soft_trans": mfloat(g1, r"\| soft_transition_density \| ([\d.]+) \|"),
            "g1_low_alt": mfloat(g1, r"\| low_alt_amplitude_rate \(seg\) \| ([\d.]+) \|"),
        })

lines = [
    "# Round-42 condition-consistency 2x2 summary",
    "",
    "Cells:",
    "",
    "- OO: oracle Stage-1 coarse + oracle C41/S4.",
    "- GO: generated Stage-1 coarse + oracle C41/S4.",
    "- OG: oracle Stage-1 coarse + generated C41/S4 from Stage-1.5.",
    "- GG: generated Stage-1 coarse + generated C41/S4 from Stage-1.5 conditioned on generated Stage-1.",
    "",
    "Interpretation:",
    "",
    "- GO bad but GG better: oracle C41/S4 is mismatched with generated Stage-1.",
    "- OG good but GG bad: Stage-1.5 is brittle to generated Stage-1.",
    "- OO good and OG good, but GO/GG bad: Stage-1 coarse is still the dominant bottleneck.",
    "- OO good, GO bad, OG bad, GG worst: both upstream distributions matter.",
    "",
    "## Headline Metrics",
    "",
    "| bucket | cell | sustained drift mean | drift p95 | track mean | LH drift | RH drift | pelvis drift | gait pred trans/s | body LW err | body RW err | G1 soft trans | low-alt rate |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        "| {bucket} | {label} | {sust_drift} | {sust_p95} | {track} | "
        "{lh} | {rh} | {pelvis} | {gait_pred_trans} | {body_lw} | "
        "{body_rw} | {g1_soft_trans} | {g1_low_alt} |".format(
            bucket=r["bucket"],
            label=r["label"],
            sust_drift=fmt(r["sust_drift"]),
            sust_p95=fmt(r["sust_p95"]),
            track=fmt(r["track"], 3),
            lh=fmt(r["lh"]),
            rh=fmt(r["rh"]),
            pelvis=fmt(r["pelvis"]),
            gait_pred_trans=fmt(r["gait_pred_trans"], 3),
            body_lw=fmt(r["body_lw"]),
            body_rw=fmt(r["body_rw"]),
            g1_soft_trans=fmt(r["g1_soft_trans"], 4),
            g1_low_alt=fmt(r["g1_low_alt"], 3),
        )
    )

lines.extend([
    "",
    "## Paths",
    "",
    f"- diag root: `{out_root / 'diag'}`",
    f"- substitute cond root: `{out_root / 'substitute_conds'}`",
])

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {summary_md}")
PY
fi

if [[ ${DRY_RUN} -eq 0 ]]; then
    log
    log "================================================================"
    log "[$(date '+%F %T')] PACK -> ${TARBALL}"
    log "================================================================"
    PACK_TARGETS=()
    [[ -d "${DIAG_ROOT}" ]] && PACK_TARGETS+=("${DIAG_ROOT}")
    [[ -f "${SUMMARY_MD}" ]] && PACK_TARGETS+=("${SUMMARY_MD}")
    [[ -f "${SUMMARY_LOG}" ]] && PACK_TARGETS+=("${SUMMARY_LOG}")
    [[ -d "${LOG_DIR}" ]] && PACK_TARGETS+=("${LOG_DIR}")
    if [[ "${PACK_NPZ}" == "1" && -d "${OUT_ROOT}/substitute_conds" ]]; then
        PACK_TARGETS+=("${OUT_ROOT}/substitute_conds")
    fi
    if [[ ${#PACK_TARGETS[@]} -eq 0 ]]; then
        log "[R42 PACK] nothing to pack"
    else
        mkdir -p "$(dirname "${TARBALL}")"
        tar -czf "${TARBALL}" "${PACK_TARGETS[@]}"
        SIZE="$(du -h "${TARBALL}" | cut -f1)"
        MANIFEST="${TARBALL%.tar.gz}_manifest.txt"
        {
            echo "# Round-42 condition 2x2 sync-back manifest"
            echo "stamp: ${STAMP}"
            echo "tarball: ${TARBALL}"
            echo "size: ${SIZE}"
            echo "pack_npz: ${PACK_NPZ}"
            echo "cells: ${CELLS[*]}"
            echo "buckets: ${BUCKETS[*]}"
            echo
            echo "## Contents"
            printf '  %s\n' "${PACK_TARGETS[@]}"
        } > "${MANIFEST}"
        log "[R42 PACK] wrote ${TARBALL} (${SIZE})"
        log "[R42 PACK] manifest ${MANIFEST}"
    fi
fi

log
log "================================================================"
log "Round-42 condition 2x2 diagnostic complete."
log "================================================================"
