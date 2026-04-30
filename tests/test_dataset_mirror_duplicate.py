from __future__ import annotations

import json

import numpy as np

from piano.data.dataset import AugmentConfig, HOIDataset


def _write_minimal_dataset(root) -> None:
    (root / "motions").mkdir(parents=True)
    (root / "objects").mkdir()
    (root / "pseudo_labels").mkdir()

    (root / "metadata.json").write_text(
        json.dumps([
            {
                "seq_id": "seq_001",
                "text": "touches the object with the left hand",
                "object_id": "obj_001",
            }
        ]),
        encoding="utf-8",
    )

    motion = np.zeros((2, 263), dtype=np.float32)
    motion[0, 259:263] = [0.1, 0.2, 0.7, 0.9]

    joints = np.zeros((2, 22, 3), dtype=np.float32)
    joints[0, 20, :] = [1.0, 2.0, 3.0]
    joints[0, 21, :] = [4.0, 5.0, 6.0]

    object_positions = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32)
    object_rotations = np.array([[0.5, 0.6, 0.7], [0.1, 0.2, 0.3]], dtype=np.float32)
    np.savez(
        root / "motions" / "seq_001.npz",
        motion_263=motion,
        joints_22=joints,
        object_positions=object_positions,
        object_rotations=object_rotations,
    )

    np.save(root / "objects" / "obj_001.npy", np.zeros((4, 3), dtype=np.float32))

    contact_state = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    contact_target_xyz_gt = np.zeros((2, 5, 3), dtype=np.float32)
    contact_target_xyz_gt[0, :, 0] = [1, 2, 3, 4, 5]
    np.savez(
        root / "pseudo_labels" / "seq_001.npz",
        contact_state=contact_state,
        contact_target_xyz_gt=contact_target_xyz_gt,
    )


def test_mirror_duplicate_doubles_dataset_and_pairs_original_with_mirror(tmp_path) -> None:
    _write_minimal_dataset(tmp_path)
    dataset = HOIDataset(
        tmp_path,
        max_seq_length=4,
        num_object_points=4,
        augment=AugmentConfig(
            enabled=True,
            mirror_prob=0.0,
            mirror_duplicate=True,
        ),
    )

    assert len(dataset) == 2
    original = dataset[0]
    mirrored = dataset[1]

    assert original["seq_id"] == mirrored["seq_id"] == "seq_001"
    assert original["text"] == "touches the object with the left hand"
    assert mirrored["text"] == "touches the object with the right hand"

    np.testing.assert_allclose(original["joints"][0, 20].numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(original["joints"][0, 21].numpy(), [4.0, 5.0, 6.0])
    np.testing.assert_allclose(mirrored["joints"][0, 20].numpy(), [-4.0, 5.0, 6.0])
    np.testing.assert_allclose(mirrored["joints"][0, 21].numpy(), [-1.0, 2.0, 3.0])

    np.testing.assert_allclose(original["motion"][0, 259:263].numpy(), [0.1, 0.2, 0.7, 0.9])
    np.testing.assert_allclose(mirrored["motion"][0, 259:263].numpy(), [0.7, 0.9, 0.1, 0.2])

    np.testing.assert_allclose(original["contact_state"][0].numpy(), [0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(mirrored["contact_state"][0].numpy(), [0.2, 0.1, 0.4, 0.3, 0.5])

    np.testing.assert_allclose(original["contact_target_xyz"][0, :, 0].numpy(), [1, 2, 3, 4, 5])
    np.testing.assert_allclose(mirrored["contact_target_xyz"][0, :, 0].numpy(), [2, 1, 4, 3, 5])

    np.testing.assert_allclose(original["object_positions"][0].numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(mirrored["object_positions"][0].numpy(), [-1.0, 2.0, 3.0])
    np.testing.assert_allclose(original["object_rotations"][0].numpy(), [0.5, 0.6, 0.7])
    np.testing.assert_allclose(mirrored["object_rotations"][0].numpy(), [0.5, -0.6, -0.7])
