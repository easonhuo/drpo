"""D-U1 task, support, and rarity metrics."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .du1_controls import normalized_excess_surprisal
from .du1_environment import CartesianSemanticEnvironment
from .du1_policy import CartesianPolicy, gather_log_probs


def evaluate(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    split: Mapping[str, torch.Tensor],
    calibration: Mapping[str, float],
) -> dict[str, float]:
    """Evaluate task, support, rarity, and cell diagnostics."""

    with torch.no_grad():
        logits, _ = model(
            split["states"],
            environment.action_embeddings,
        )
        log_probabilities = F.log_softmax(
            logits,
            dim=-1,
        )
        probabilities = log_probabilities.exp()
        expected_reward = (probabilities * split["reward_matrix"]).sum(1).mean()
        hidden_probability = (
            probabilities.gather(
                1,
                split["hidden_optimal_actions"],
            )
            .sum(1)
            .mean()
        )
        positive_probabilities = gather_log_probs(
            log_probabilities,
            split["positive_pairs"],
        ).exp()
        positive_probability = positive_probabilities.sum(-1).sum(-1).mean()
        entropy = -(probabilities * log_probabilities).sum(1)
        action_support = entropy.exp()
        observed_probabilities = probabilities[
            :,
            : environment.observed_action_count,
        ]
        observed_family = observed_probabilities.reshape(
            -1,
            environment.prototype_count,
            2,
        ).sum(-1)
        hidden_family = probabilities[
            :,
            environment.observed_action_count :,
        ]
        family_probabilities = torch.cat(
            [observed_family, hidden_family],
            dim=1,
        )
        family_entropy = -(
            family_probabilities * family_probabilities.clamp_min(1.0e-12).log()
        ).sum(1)
        prototype_support = family_entropy.exp()
        common_mass = observed_probabilities[
            :,
            0::2,
        ].sum(1)
        rare_mass = observed_probabilities[:, 1::2].sum(1) + hidden_family.sum(1)
        rarity_coordinate = model.rarity_coordinate(split["states"])
        result = {
            "expected_semantic_reward": float(expected_reward),
            "hidden_optimal_family_probability": float(hidden_probability),
            "positive_support_probability": float(positive_probability),
            "action_entropy_mean": float(entropy.mean()),
            "action_effective_support": float(action_support.mean()),
            "prototype_entropy_mean": float(family_entropy.mean()),
            "prototype_effective_support": float(prototype_support.mean()),
            "common_total_probability": float(common_mass.mean()),
            "rare_total_probability": float(rare_mass.mean()),
            "rarity_logit_gap_mean": float((2.0 * rarity_coordinate.abs()).mean()),
        }

        useful_pair = gather_log_probs(
            log_probabilities,
            split["useful_pair"],
        )
        unhelpful_pair = gather_log_probs(
            log_probabilities,
            split["unhelpful_pair"],
        )
        dynamic = {
            "useful_common": useful_pair.max(1).values,
            "useful_rare": useful_pair.min(1).values,
            "unhelpful_common": (unhelpful_pair.max(1).values),
            "unhelpful_rare": (unhelpful_pair.min(1).values),
        }
        for cell, log_probability in dynamic.items():
            result[f"{cell}_surprisal_mean"] = float((-log_probability).mean())
            result[f"{cell}_probability_mean"] = float(log_probability.exp().mean())
            result[f"{cell}_normalized_excess_mean"] = float(
                normalized_excess_surprisal(
                    log_probability,
                    calibration,
                ).mean()
            )
        return result
