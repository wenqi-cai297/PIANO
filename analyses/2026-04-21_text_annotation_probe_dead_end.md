# Text / Annotation Probe — Stricter Prior Dead End — 2026-04-21

## Context

After the plain threshold sweep (2026-04-20) gave us per-(threshold,
body part) `frame_rate` and `seq_reached` curves, picking thresholds
still relied on a soft "does this match my intuition about the
dataset" prior. Following the question of where that intuition comes
from, the plan was to upgrade the prior: use InterAct's `text.txt`
`start/end` timestamps (HumanML3D convention) to partition each
sequence into "inside action window" vs "outside" and measure whether
contact labels fire in the right place.

Tool `piano-action-segment-sweep` was built on that assumption
(commit `dde931c`) and returned `0 / N parseable` on every subset.
Commit `fd88445` added `piano-probe-text-annotations` to dump text
and annotation files verbatim. This note records what the probe
found and why the stricter-prior path was abandoned.

## Findings

### text.txt timestamps are placeholder

All 4 subsets' `text.txt` follows structure
`caption#pos_tags#start#end`, but `start` and `end` are always
`0.0#0.0`. Verified on 12 sampled text.txt files across subsets.
Example from `omomo_correct_v2/sub10_clothesstand_000/text.txt`:

```
Lift the clothesstand, move the clothesstand, and put down the
clothesstand.#lift/VERB the/DET clothesstand/NOUN ... #0.0#0.0
```

chairs / imhd / neuraldome are the same — `0.0#0.0` in every line.
`action_segment_sweep`'s parser rejects `end <= start`, so the 0/N
parseable rate was correct behaviour, not a parsing bug.

### Real action windows live in per-dataset CSVs

`<InterAct>/annotation/<kind>/<subset>.csv` for
`kind ∈ {action, natural, raw, change, shorten}` is the annotation
root. Row schema:

```
Name, AssignmentStatus, Input.video_url, Answer.Behave, Answer.action
```

For `kind ∈ {natural, raw, change, shorten}`, `Answer.action` is a
multi-line string where each line is

```
<caption>#<start_frame>#<end_frame>
```

at **source fps (30)**. Multiple captions per row, separated by `\n`
(natural/change/shorten) or `\r\n` (raw).

`kind = action` is categorical ("Sit", "Move", "Swing") with no
frame ranges.

### Two structural problems with using the CSVs

**(1) omomo_correct_v2 has no frame-range CSV.** Every kind except
`action/` is missing `omomo_correct_v2.csv`. The `action/` one only
has categorical labels. 4890 sequences (58% of our data) get no
action-window signal.

**(2) video_url → seq_id mapping is non-trivial for every subset.**
CSV video filenames and our preprocessed `seq_id` names don't match
directly:

| subset | CSV video_url example | Our seq_id example |
|---|---|---|
| chairs | `CHAIRS_Sub0853_Obj68_Seg0.mp4` | `Sub0001_Obj116_Seg0_0` / `_300` / `_600` |
| imhd | `IMHD_20230825_songzn_bat_bat_twoends_rotate_0_baseball.mp4` | `20230825_songzn_bat_bat_holdhandle_hit_0_0` |
| neuraldome | `NEURALDOME_subject01_table_table.mp4` | `subject01_baseball_0` |

chairs has the clearest pattern: one CSV row per original sequence,
and our `seq_id` is `<stripped-video-name>_<frame_offset>`. Frame
offset (0, 300, 600, ...) is the start of a 300-frame window cut
from the original. So to use the CSV, we'd have to:

1. Strip `CHAIRS_` / `.mp4` from the video_url → match our
   `Sub0853_Obj68_Seg0` prefix.
2. For each caption+frame-range in the CSV row, shift frames by
   `-frame_offset` and clip to `[0, 300)`.
3. Convert source-fps (30) frame indices to target-fps (20).

imhd / neuraldome are messier. Row counts (164 / 180) are much
smaller than our seq counts (592 / 1491), so either many sequences
are unannotated or each CSV row covers many segments. The imhd
example also encodes the action in the filename (`bat_rotate` vs
`bat_holdhandle_hit`) which doesn't match our seq_id — might be
different capture sessions annotated as one video, or the CSV uses
a different naming.

## Decision: drop the stricter-prior path

Cost / benefit:

- **Cost**: ~1 engineer-day to build CSV parsing + per-subset
  video→seq mapping + frame-offset arithmetic + cross-fps conversion.
  Another sweep run to apply the new parser. Ongoing maintenance as
  annotation format might shift.
- **Benefit**: covers 42% of the data (chairs + imhd + neuraldome);
  omomo (58%) still needs a different validation signal. Produces a
  sharper threshold-selection prior than plain seq_reached curves,
  but still a proxy for the actual downstream criterion
  (predictor + generator performance).

We already have the thresholds (`d641732`: hand 0.08 / foot 0.06 /
pelvis 0.20) backed by three independent arguments:

1. Anatomy: SMPL joint-to-skin-surface offsets match the thresholds
   to within ±0.02 m.
2. Plain sweep `seq_reached` for chairs-pelvis saturates at 93% by
   threshold 0.20 — this is the elbow of the curve.
3. Data ranges from the raw distance distribution (chairs pelvis
   p10 = 10 cm, p50 = 18 cm) are consistent with the interpretation
   that real sitting contact produces joint-to-mesh distances of
   12-20 cm.

The next-ring validation is not a sharper numerical prior, it is:

- Watching rendered pseudo-label videos (`piano-visualize-pseudo-labels`).
- The quality_flags output from the rich-stats pass (already
  implemented in `run_all.py`).
- Stage A predictor held-out accuracy.

None of those require the action window.

## Artefacts kept

- `piano-probe-text-annotations` (`fd88445`) — useful probe even if
  we never consume CSVs programmatically. Leaves `preview.md` and
  `summary.json` on disk for future reference.
- `piano-action-segment-sweep` (`dde931c`) — skeleton left in place.
  If someone re-opens this path (e.g. to do Option B for chairs
  only), the structure is already there and only the parser /
  mapping need rewriting.
- `runs/checks/text_annotations/<ts>/preview.md` — the verbatim dump
  that would need to be re-read to re-derive these conclusions.

## Action Items (→ PLAN.md)

- [x] Drop "action-segment sweep" as a gate for v2 rerun.
- [ ] v2 rerun using `d641732` thresholds remains the next step.
- [ ] Post-v2, rely on `quality_flags` + visualization for validation
      before Stage A training.
- [ ] If Stage A held-out accuracy < 70% and `sitting` fraction
      looks off, revisit Option B (chairs CSV action window) as a
      targeted repair — not as a gate.
