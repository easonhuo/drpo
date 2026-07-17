"""D-U1 task, support, and environment-validity diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .du1_controls import normalized_excess_surprisal
from .du1_environment import CartesianSemanticEnvironment
from .du1_policy import (
    CartesianPolicy,
    cell_log_probs,
    gather_log_probs,
)
from .du1_protocol import CELL_NAMES, DU1Protocol


def _oracle_metrics(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    split: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    logits, _ = model(
        split["states"],
        environment.action_embeddings,
    )
    probabilities = F.softmax(logits, dim=-1)
    per_state_reward = (
        probabilities * split["reward_matrix"]
    ).sum(1)
    valid: list[float] = []
    for cell in CELL_NAMES:
        pair = (
            split["useful_pair"]
            if cell.startswith("useful")
            else split["unhelpful_pair"]
        )
        pair_probabilities = probabilities.gather(1, pair)
        if cell.endswith("common"):
            selected = pair_probabilities.argmax(
                dim=1,
                keepdim=True,
            )
        else:
            selected = pair_probabilities.argmin(
                dim=1,
                keepdim=True,
            )
        dynamic_action = pair.gather(
            1,
            selected,
        ).squeeze(1)
        action_probability = probabilities.gather(
            1,
            dynamic_action[:, None],
        ).squeeze(1)
        action_reward = split["reward_matrix"].gather(
            1,
            dynamic_action[:, None],
        ).squeeze(1)
        effect = action_probability * (
            per_state_reward - action_reward
        )
        valid.append(
            float(
                (
                    effect > 0
                    if cell.startswith("useful")
                    else effect < 0
                )
                .float()
                .mean()
            )
        )

    probe = environment.protocol.rarity_shift_probe
    shifted_probabilities = F.softmax(
        logits
        + probe
        * model.action_rarity_sign[None, :],
        dim=-1,
    )
    base_reward = per_state_reward.mean()
    shifted_reward = (
        shifted_probabilities * split["reward_matrix"]
    ).sum(1).mean()
    hidden = split["hidden_optimal_actions"]
    base_hidden = probabilities.gather(
        1,
        hidden,
    ).sum(1).mean()
    shifted_hidden = shifted_probabilities.gather(
        1,
        hidden,
    ).sum(1).mean()
    return {
        "utility_oracle_sign_valid_fraction": min(valid),
        "counterfactual_common_shift_reward_delta": float(
            shifted_reward - base_reward
        ),
        "counterfactual_common_shift_hidden_probability_delta": (
            float(shifted_hidden - base_hidden)
        ),
    }


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
        expected_reward = (
            probabilities * split["reward_matrix"]
        ).sum(1).mean()
        hidden_probability = probabilities.gather(
            1,
            split["hidden_optimal_actions"],
        ).sum(1).mean()
        positive_probabilities = gather_log_probs(
            log_probabilities,
            split["positive_pairs"],
        ).exp()
        positive_probability = (
            positive_probabilities.sum(-1).sum(-1).mean()
        )
        entropy = -(
            probabilities * log_probabilities
        ).sum(1)
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
            family_probabilities
            * family_probabilities.clamp_min(1.0e-12).log()
        ).sum(1)
        prototype_support = family_entropy.exp()
        common_mass = observed_probabilities[
            :,
            0::2,
        ].sum(1)
        rare_mass = (
            observed_probabilities[:, 1::2].sum(1)
            + hidden_family.sum(1)
        )
        rarity_coordinate = model.rarity_coordinate(
            split["states"]
        )
        result = {
            "expected_semantic_reward": float(expected_reward),
            "hidden_optimal_family_probability": float(
                hidden_probability
            ),
            "positive_support_probability": float(
                positive_probability
            ),
            "action_entropy_mean": float(entropy.mean()),
            "action_effective_support": float(
                action_support.mean()
            ),
            "prototype_entropy_mean": float(
                family_entropy.mean()
            ),
            "prototype_effective_support": float(
                prototype_support.mean()
            ),
            "common_total_probability": float(
                common_mass.mean()
            ),
            "rare_total_probability": float(
                rare_mass.mean()
            ),
            "rarity_logit_gap_mean": float(
                (2.0 * rarity_coordinate.abs()).mean()
            ),
            "rarity_coordinate_positive_fraction": float(
                (rarity_coordinate > 0).float().mean()
            ),
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
            "unhelpful_common": (
                unhelpful_pair.max(1).values
            ),
            "unhelpful_rare": (
                unhelpful_pair.min(1).values
            ),
        }
        for cell, log_probability in dynamic.items():
            result[f"{cell}_surprisal_mean"] = float(
                (-log_probability).mean()
            )
            result[f"{cell}_probability_mean"] = float(
                log_probability.exp().mean()
            )
            result[
                f"{cell}_normalized_excess_mean"
            ] = float(
                normalized_excess_surprisal(
                    log_probability,
                    calibration,
                ).mean()
            )
        result.update(
            _oracle_metrics(
                model,
                environment,
                split,
            )
        )
        return result


def policy_geometry_audit(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    protocol: DU1Protocol,
) -> dict[str, Any]:
    """Audit the utility and rarity separation required by D-U1."""

    count = min(protocol.audit_states, environment.train_count)
    index = torch.arange(count)
    positive_log_probability, cells, _ = cell_log_probs(
        model,
        environment,
        environment.train,
        index,
    )
    rarity_parameters = tuple(
        model.rarity_residual_head.parameters()
    )
    positive_gradients = torch.autograd.grad(
        -positive_log_probability.mean(),
        rarity_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    positive_norm = math.sqrt(
        sum(
            float(
                gradient.detach()
                .double()
                .square()
                .sum()
            )
            for gradient in positive_gradients
            if gradient is not None
        )
    )
    norms: dict[str, float] = {}
    for index_cell, cell in enumerate(CELL_NAMES):
        gradients = torch.autograd.grad(
            cells[cell].mean(),
            rarity_parameters,
            retain_graph=index_cell < len(CELL_NAMES) - 1,
            allow_unused=True,
        )
        norms[cell] = math.sqrt(
            sum(
                float(
                    gradient.detach()
                    .double()
                    .square()
                    .sum()
                )
                for gradient in gradients
                if gradient is not None
            )
        )
    useful_ratio = (
        norms["useful_rare"]
        / max(norms["useful_common"], 1.0e-12)
    )
    unhelpful_ratio = (
        norms["unhelpful_rare"]
        / max(norms["unhelpful_common"], 1.0e-12)
    )
    with torch.no_grad():
        split = {
            key: (
                value[:count]
                if isinstance(value, torch.Tensor)
                and value.shape[0] == environment.train_count
                else value
            )
            for key, value in environment.train.items()
        }
        oracle = _oracle_metrics(
            model,
            environment,
            split,
        )
    passed = bool(
        positive_norm <= 1.0e-6
        and useful_ratio >= 5.0
        and unhelpful_ratio >= 5.0
        and oracle[
            "utility_oracle_sign_valid_fraction"
        ]
        >= protocol.utility_sign_fraction_min
        and -oracle[
            "counterfactual_common_shift_reward_delta"
        ]
        >= protocol.minimum_rarity_shift_reward_drop
        and -oracle[
            "counterfactual_common_shift_hidden_probability_delta"
        ]
        >= protocol.minimum_rarity_shift_hidden_probability_drop
    )
    return {
        "passed": passed,
        "positive_rarity_gradient_norm": positive_norm,
        "cell_shared_rarity_gradient_norms": norms,
        "useful_rare_to_common_shared_rarity_gradient_ratio": (
            useful_ratio
        ),
        "unhelpful_rare_to_common_shared_rarity_gradient_ratio": (
            unhelpful_ratio
        ),
        "utility_oracle_sign_valid_fraction": oracle[
            "utility_oracle_sign_valid_fraction"
        ],
        "rarity_shift_reward_drop": -oracle[
            "counterfactual_common_shift_reward_delta"
        ],
        "rarity_shift_hidden_probability_drop": -oracle[
            "counterfactual_common_shift_hidden_probability_delta"
        ],
        "trainable_per_action_bias": False,
    }
