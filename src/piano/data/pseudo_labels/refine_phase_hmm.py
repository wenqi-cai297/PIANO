"""Refine interaction phase labels using a Hidden Markov Model.

The heuristic phase labels from ``extract_phase`` can be noisy.  This
module fits an HMM to smooth the phase sequence, encouraging temporally
consistent transitions while respecting the observed features.

The HMM uses per-frame features:
    - hand-object distance
    - hand contact score (max of left/right)
    - object velocity magnitude

and is initialized from the heuristic phase labels so it converges
quickly to a refined labeling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piano.data.pseudo_labels.extract_phase import NUM_PHASES


@dataclass(slots=True)
class HMMConfig:
    """Configuration for HMM phase refinement."""

    n_components: int = NUM_PHASES   # one state per phase
    n_iter: int = 50                 # EM iterations
    covariance_type: str = "diag"
    random_state: int = 42


def refine_phases_hmm(
    features: np.ndarray,
    initial_phases: np.ndarray,
    config: HMMConfig | None = None,
) -> np.ndarray:
    """Refine phase labels using a Gaussian HMM.

    Parameters
    ----------
    features : (T, D) — per-frame feature matrix.  Typically D=3:
        [hand_obj_distance, hand_contact_score, object_velocity].
    initial_phases : (T,) — heuristic phase labels (used for initialization)
    config : HMM parameters

    Returns
    -------
    refined_phases : (T,) — HMM-smoothed phase labels
    """
    from hmmlearn.hmm import GaussianHMM

    if config is None:
        config = HMMConfig()

    T, D = features.shape

    # --- Initialize HMM parameters from heuristic labels ---
    hmm = GaussianHMM(
        n_components=config.n_components,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        random_state=config.random_state,
        init_params="",  # we set all params manually
    )

    # Start probability: fraction of sequences starting in each phase
    startprob = np.zeros(config.n_components)
    startprob[initial_phases[0]] = 1.0
    hmm.startprob_ = startprob

    # Transition matrix: count transitions in heuristic labels, add smoothing
    transmat = np.ones((config.n_components, config.n_components)) * 0.01
    for t in range(1, T):
        transmat[initial_phases[t - 1], initial_phases[t]] += 1.0
    # Normalize rows
    transmat /= transmat.sum(axis=1, keepdims=True)
    hmm.transmat_ = transmat

    # Emission means and variances from heuristic labels
    means = np.zeros((config.n_components, D))
    covars = np.ones((config.n_components, D))
    for k in range(config.n_components):
        mask = initial_phases == k
        if mask.sum() > 1:
            means[k] = features[mask].mean(axis=0)
            covars[k] = features[mask].var(axis=0) + 1e-4
        else:
            means[k] = features.mean(axis=0)
            covars[k] = features.var(axis=0) + 1e-4
    hmm.means_ = means
    hmm.covars_ = covars

    # --- Fit and predict ---
    hmm.fit(features)
    refined_phases = hmm.predict(features)

    return refined_phases.astype(np.int64)


def build_phase_features(
    joints: np.ndarray,
    contact_state: np.ndarray,
    object_positions: np.ndarray | None = None,
    fps: float = 30.0,
) -> np.ndarray:
    """Build the feature matrix for HMM refinement.

    Parameters
    ----------
    joints : (T, 22, 3)
    contact_state : (T, 5)
    object_positions : (T, 3) or None — object center per frame
    fps : frame rate

    Returns
    -------
    features : (T, 3) — [hand_obj_dist, hand_contact, obj_velocity]
    """
    from piano.utils.smpl_utils import BODY_PART_INDICES

    T = len(joints)

    # Hand-object distance (minimum of left and right hand)
    left_hand = joints[:, BODY_PART_INDICES[0], :]
    right_hand = joints[:, BODY_PART_INDICES[1], :]

    if object_positions is None:
        # Estimate from mean contact position
        hand_contact = np.maximum(contact_state[:, 0], contact_state[:, 1])
        contact_mask = hand_contact > 0.5
        if contact_mask.any():
            mean_pos = ((left_hand[contact_mask] + right_hand[contact_mask]) / 2).mean(axis=0)
            object_positions = np.tile(mean_pos, (T, 1))
        else:
            object_positions = np.zeros((T, 3))
    elif object_positions.ndim == 1:
        object_positions = np.tile(object_positions, (T, 1))

    dist_left = np.linalg.norm(left_hand - object_positions, axis=-1)
    dist_right = np.linalg.norm(right_hand - object_positions, axis=-1)
    hand_obj_dist = np.minimum(dist_left, dist_right)

    # Hand contact score (max of left/right)
    hand_contact = np.maximum(contact_state[:, 0], contact_state[:, 1])

    # Object velocity
    obj_vel = np.zeros(T)
    obj_vel[1:] = np.linalg.norm(np.diff(object_positions, axis=0), axis=-1) * fps

    features = np.stack([hand_obj_dist, hand_contact, obj_vel], axis=-1)  # (T, 3)
    return features
