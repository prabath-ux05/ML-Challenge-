"""Pairwise matching models and classification decision engines."""

from business_entity_resolution.matching.decision import (
    ThresholdOptimizer,
    ThresholdSweepRow,
    predict_matches_for_entities,
)
from business_entity_resolution.matching.model import EntityMatchClassifier
from business_entity_resolution.matching.train import EntityMatchTrainer
from business_entity_resolution.matching.tune import ModelTuner

__all__ = [
    "EntityMatchClassifier",
    "ThresholdOptimizer",
    "ThresholdSweepRow",
    "predict_matches_for_entities",
    "EntityMatchTrainer",
    "ModelTuner",
]
