# OMOMO (CHOIS processed_data) Format Inspection — 2026-04-19

## Context

After downloading the CHOIS `processed_data.tar.gz` (8.6 GB from Google Drive)
into `/media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/`, we needed to
confirm the on-disk format matches our preprocessing assumptions before
writing the OMOMO → HumanML3D conversion.

- Runner: `scripts/server/check_omomo_format.sh` wrapping `piano-check-omomo`

## Results

```
Dataset contents: /media/gpu-server-1/4TB_for_data/Cai/datasets/omomo/processed_data
- Train sequences (pickle): 5280  — sub1..sub15
- Test sequences (pickle):   602  — sub16..sub17
- Object meshes (.obj):       15   (captured_objects/)
- Rest-pose meshes (.ply):    13   (rest_object_geo/; vacuum, mop missing)
- Text annotation JSONs:    4912
- Contact label files:      5882
```

Per-sequence dict keys (first sample: `sub10_clothesstand_000`, 132 frames):

| key | shape | dtype | notes |
|-----|-------|-------|-------|
| `seq_name` | str | — | `sub{N}_{object}_{take}` |
| `betas` | (1, 16) | float32 | 16 SMPL-X shape coefficients (not the usual 10) |
| `gender` | scalar | `<U6` | 'male' / 'female' |
| `trans` | (T, 3) | float32 | root translation, world frame |
| `root_orient` | (T, 3) | float32 | axis-angle |
| `pose_body` | (T, 63) | float64 | 21 body joints axis-angle |
| `obj_trans` | (T, 3, 1) | float32 | **trailing axis=1**, index `[..., 0]` |
| `obj_rot` | (T, 3, 3) | float32 | rotation matrix |
| `obj_scale` | (T,) | float32 | per-frame scale |
| `obj_com_pos` | (T, 3) | float32 | object center of mass |
| `trans2joint` | (3,) | float32 | offset between SMPL trans and root joint |
| `rest_offsets` | (24, 3) | float32 | T-pose joint offsets |

First 20 sequence lengths: `[132, 119, 137, 143, 128, 128, 138, 126, 135, 157,
122, 144, 180, 166, 126, 196, 148, 203, 170, 211]` — i.e., 4.4s–7s clips at 30fps.

Sample text annotation: `"Lift the clothesstand, move the clothesstand, and
put down the clothesstand."`

Contact label sample shape: `(132, 4)`, int64, range {0, 1} — binary flags for
`[left_hand, right_hand, left_foot, right_foot]`.

## Observations

Several format surprises worth noting:

1. **`.p` files are joblib pickles, not vanilla pickle.** Opening them with
   `pickle.load` fails (`UnpicklingError: invalid load key 'A'`). Switched
   to `joblib.load` in the inspection script and added `joblib` to deps.

2. **16 betas, not 10.** SMPL-X uses `num_betas=16` in this data — the
   loading code for SMPL-X model must specify this.

3. **`obj_trans` has a trailing singleton axis.** Code must do
   `obj_trans[..., 0]` to get the usable `(T, 3)` array.

4. **vacuum / mop have no rest-pose mesh.** The `rest_object_geo/` folder
   only has 13 `.ply` files (clothesstand, floorlamp, largebox, largetable,
   monitor, plasticbox, smallbox, smalltable, suitcase, trashcan, tripod,
   whitechair, woodchair). CHOIS's own preprocessing code skips vacuum/mop
   because they are two-part articulated objects — their handling needs a
   different data pipeline (separate top/bottom rigid bodies).

5. **Text coverage is partial** but very high: 4912 annotations for 5882
   sequences ≈ 83.5% at source (after we filter vacuum/mop the coverage
   rises to 98.4%).

6. **No hand/face pose stored.** Only body (21 joints) — consistent with
   OMOMO focus on whole-body object interaction, not fine manipulation.

7. **CHOIS >> raw OMOMO for our purposes.** The CHOIS preprocessed bundle
   already unifies the text annotations, contact labels, and rest-pose
   geometries — we would otherwise have to stitch these from the raw OMOMO
   repo + `omomo_text_anno.zip`.

## Diagnosis

The format is clean and complete enough for our preprocessing. The main
design decision is what to do with vacuum/mop — we keep CHOIS's default
and skip them, documented as `PreprocessConfig.skip_objects` in
`preprocess_omomo.py`. This is inherited behavior, not a PIANO limitation.

## Implications

- Our preprocessing needs SMPL-X FK (not raw joint coords) because the
  data only ships parameters. Requires `smplx` Python package + SMPL-X
  `.npz` model files (downloaded separately).
- FPS mismatch: data is 30fps, MoMask's HumanML3D format is 20fps. Need
  linear interpolation during preprocessing.
- `gender` field is a numpy scalar (`<U6`); `str(...)` works but the
  adapter must handle unknown values (defaulted to "male").

## Action Items (→ PLAN.md)

- [x] Write `preprocess_omomo.py` using SMPL-X FK
- [x] Add `joblib` to dependencies
- [x] Default `skip_objects=("vacuum", "mop")`
- [ ] Flag articulated-object handling as future work
