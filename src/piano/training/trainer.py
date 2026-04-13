"""Shared Accelerate-based training loop.

Provides the common training infrastructure used by all three training stages
(predictor, generator, joint finetune). Each stage script configures its own
model, loss, and data, then delegates to the shared loop here.
"""
from __future__ import annotations


def main() -> None:
    """CLI entrypoint for ``piano-train``."""
    raise NotImplementedError("Training not yet implemented — see train_predictor.py")


if __name__ == "__main__":
    main()
