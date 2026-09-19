"""D-U1 revision-4 categorical policy and log-probability primitives."""

from __future__ import annotations

import copy
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .du1_environment import CartesianSemanticEnvironment, unit
from .du1_protocol import DU1Protocol


class CartesianPolicy(nn.Module):
    def __init__(
        self,
        protocol: DU1Protocol,
        environment: CartesianSemanticEnvironment,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(protocol.state_dim, protocol.hidden_dim),
            nn.Tanh(),
            nn.Linear(protocol.hidden_dim, protocol.hidden_dim),
            nn.Tanh(),
        )
        self.direction_head = nn.Linear(
            protocol.hidden_dim,
            protocol.semantic_dim,
        )
        self.rarity_residual_head = nn.Linear(
            protocol.hidden_dim,
            1,
        )
        nn.init.zeros_(self.rarity_residual_head.weight)
        nn.init.zeros_(self.rarity_residual_head.bias)
        self.reference_trunk = copy.deepcopy(self.trunk)
        self.reference_direction_head = copy.deepcopy(self.direction_head)
        for parameter in self.reference_trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in self.reference_direction_head.parameters():
            parameter.requires_grad_(False)
        self.fixed_concentration = protocol.fixed_concentration
        self.initial_rarity_half_gap = environment.initial_rarity_half_gap()
        self.register_buffer(
            "action_rarity_sign",
            environment.action_rarity_sign.clone().float(),
        )

    def semantic_residual(
        self,
        states: torch.Tensor,
        action_embeddings: torch.Tensor,
        reference_direction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(states)
        direction = unit(self.direction_head(features))
        if reference_direction is None:
            with torch.no_grad():
                reference_direction = unit(
                    self.reference_direction_head(self.reference_trunk(states))
                )
        residual = self.fixed_concentration * (
            (direction - reference_direction) @ action_embeddings.T
        )
        return residual, direction, features

    def forward(
        self,
        states: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_logits, direction, features = self.semantic_residual(
            states,
            action_embeddings,
        )
        rarity_coordinate = self.initial_rarity_half_gap + self.rarity_residual_head(
            features
        ).squeeze(-1)
        logits = semantic_logits + rarity_coordinate[:, None] * self.action_rarity_sign[None, :]
        return logits, direction

    def rarity_coordinate(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        features = self.trunk(states)
        return self.initial_rarity_half_gap + self.rarity_residual_head(features).squeeze(-1)


def trainable_parameters(
    model: nn.Module,
) -> tuple[nn.Parameter, ...]:
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def cache_reference_directions(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
) -> None:
    with torch.no_grad():
        for split in (environment.train, environment.test):
            split["reference_direction"] = unit(
                model.reference_direction_head(model.reference_trunk(split["states"]))
            )


def batch_indices(
    seed: int,
    step: int,
    count: int,
    batch_size: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(
        900_000_003 + int(seed) * 100_003 + int(step)
    )
    return torch.randint(
        0,
        count,
        (batch_size,),
        generator=generator,
    )


def gather_log_probs(
    log_probs: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if actions.ndim == 1:
        return log_probs.gather(
            1,
            actions[:, None],
        ).squeeze(1)
    flat = actions.reshape(actions.shape[0], -1)
    return log_probs.gather(
        1,
        flat,
    ).reshape(actions.shape)


def cell_log_probs(
    model: CartesianPolicy,
    environment: CartesianSemanticEnvironment,
    split: Mapping[str, torch.Tensor],
    index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    states = split["states"][index]
    reference = split.get("reference_direction")
    reference_batch = None if reference is None else reference[index]
    semantic_logits, _, features = model.semantic_residual(
        states,
        environment.action_embeddings,
        reference_batch,
    )
    rarity_residual = model.rarity_residual_head(features).squeeze(-1)
    rarity_coordinate = model.initial_rarity_half_gap + rarity_residual
    logits = semantic_logits + rarity_coordinate[:, None] * model.action_rarity_sign[None, :]
    log_probs = F.log_softmax(logits, dim=-1)
    prototype_logits = semantic_logits[
        :,
        : environment.observed_action_count : 2,
    ]
    prototype_log_probs = F.log_softmax(
        prototype_logits,
        dim=-1,
    )
    positive = gather_log_probs(
        prototype_log_probs,
        split["positive_proto"][index],
    ).mean(1)
    useful_pair = gather_log_probs(
        log_probs,
        split["useful_pair"][index],
    )
    unhelpful_pair = gather_log_probs(
        log_probs,
        split["unhelpful_pair"][index],
    )
    cells = {
        "useful_common": useful_pair.max(dim=1).values,
        "useful_rare": useful_pair.min(dim=1).values,
        "unhelpful_common": (unhelpful_pair.max(dim=1).values),
        "unhelpful_rare": (unhelpful_pair.min(dim=1).values),
    }
    return positive, cells, rarity_residual
