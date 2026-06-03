# PIANO

**PIANO** is a research-oriented Python codebase for object-conditioned human motion generation.
It explores structured intermediate representations for improving the plausibility and controllability of human-object interaction motion.

This public repository is intended to provide a cleaned implementation scaffold, reusable utilities, and minimal runnable entry points. Detailed experimental protocols, full ablation records, and publication-specific results will be released separately when appropriate.

## Overview

Human-object interaction generation requires a model to synthesize full-body motion that is consistent with both language-level intent and object geometry. PIANO focuses on this problem from a structured-generation perspective: rather than treating motion generation as a single opaque mapping, the project organizes motion synthesis around intermediate representations that make trajectory, interaction, and pose-level control easier to inspect and evaluate.

The repository currently includes:

- Dataset and preprocessing utilities for human-object interaction motion data.
- Model components for object-conditioned motion generation.
- Training entry points based on PyTorch and HuggingFace Accelerate.
- Sampling and visualization utilities for generated motion.
- Tests for core data, model, and loss components.

The codebase is under active research development. Interfaces may change before the first stable release.

## Repository layout

```text
src/piano/
  data/          Dataset loading, preprocessing, pseudo-label utilities, geometry helpers
  models/        Motion-generation models and conditioning modules
  training/      Training entry points and loss utilities
  inference/     Sampling, visualization, and generated-motion utilities
  evaluation/    Evaluation helpers and diagnostic scripts
  checks/        Consistency checks and validation utilities

scripts/
  prep/          Data preparation scripts
  eval/          Evaluation launchers
  vis/           Visualization helpers
  checks/        Repository and experiment sanity checks

configs/
  training/      Example training configurations

tests/           Pytest-based unit and integration tests
