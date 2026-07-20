"""D-U1 revision-4 utility×rarity environment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .du1_protocol import CELL_NAMES, DU1Protocol


def unit(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, p=2, dim=-1, eps=1.0e-12)


class CartesianSemanticEnvironment:
    """Exact observed utility×rarity lattice plus hidden rare actions."""

    def __init__(self, protocol: DU1Protocol, seed: int):
        self.protocol = protocol
        self.seed = int(seed)
        self.state_dim = protocol.state_dim
        self.semantic_dim = protocol.semantic_dim
        self.prototype_count = protocol.semantic_prototypes
        self.hidden_prototype_count = protocol.hidden_semantic_prototypes
        self.semantic_family_count = self.prototype_count + self.hidden_prototype_count
        self.rarity_replicas = protocol.rarity_replicas
        self.observed_action_count = protocol.observed_action_count
        self.hidden_action_count = protocol.hidden_action_count
        self.action_count = protocol.action_count
        self.hidden_optimal_count = protocol.hidden_optimal_actions_per_state
        self.n_positive = protocol.positive_prototypes_per_state
        self.train_count = protocol.train_states
        self.test_count = protocol.test_states
        self.target_offset = protocol.target_offset
        self.positive_advantage = protocol.positive_advantage
        self.negative_advantage = protocol.negative_advantage
        self.neutral_reward = protocol.neutral_observed_reward
        self.positive_reward = protocol.positive_observed_reward
        self.useful_reward = protocol.useful_negative_reward
        self.unhelpful_reward = protocol.unhelpful_negative_reward
        self.hidden_reward_min = protocol.hidden_reward_min
        self.hidden_reward_max = protocol.hidden_reward_max

        generator = torch.Generator(device="cpu").manual_seed(410_003 + self.seed)
        half = self.prototype_count // 2
        base = unit(torch.randn(half, self.semantic_dim, generator=generator))
        observed = torch.cat([base, -base], dim=0)
        observed = observed[torch.randperm(self.prototype_count, generator=generator)].contiguous()
        hidden = unit(
            torch.randn(
                self.hidden_prototype_count,
                self.semantic_dim,
                generator=generator,
            )
        )
        self.prototype_embeddings = torch.cat([observed, hidden], dim=0)

        observed_action_proto = torch.arange(self.prototype_count).repeat_interleave(2)
        hidden_action_proto = torch.arange(
            self.prototype_count,
            self.semantic_family_count,
        )
        self.action_prototype = torch.cat([observed_action_proto, hidden_action_proto])
        observed_rarity = torch.tensor(
            [0, 1],
            dtype=torch.long,
        ).repeat(self.prototype_count)
        hidden_rarity = torch.ones(
            self.hidden_action_count,
            dtype=torch.long,
        )
        self.action_rarity = torch.cat([observed_rarity, hidden_rarity])
        self.action_rarity_sign = torch.where(
            self.action_rarity == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        )
        self.action_embeddings = self.prototype_embeddings[self.action_prototype]

        geometry = torch.Generator(device="cpu").manual_seed(420_003 + self.seed)
        self.w_plus = torch.randn(
            self.state_dim,
            self.semantic_dim,
            generator=geometry,
        )
        self.w_direction = torch.randn(
            self.state_dim,
            self.semantic_dim,
            generator=geometry,
        )
        self.train = self._build_split(
            self.train_count,
            430_003 + self.seed,
        )
        self.test = self._build_split(
            self.test_count,
            440_003 + self.seed,
        )

    @staticmethod
    def action_id(prototype: torch.Tensor, rarity: int) -> torch.Tensor:
        return prototype * 2 + int(rarity)

    @staticmethod
    def _topk_excluding(
        scores: torch.Tensor,
        banned: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        masked = scores.masked_fill(banned, -torch.inf)
        values, indices = masked.topk(count, dim=1)
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("insufficient admissible prototypes")
        return indices

    def _build_split(
        self,
        count: int,
        split_seed: int,
    ) -> dict[str, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(split_seed)
        states = torch.randn(
            count,
            self.state_dim,
            generator=generator,
        )
        t_plus = unit(states @ self.w_plus)
        raw = states @ self.w_direction
        raw = raw - (raw * t_plus).sum(-1, keepdim=True) * t_plus
        weak = raw.norm(dim=-1) < 1.0e-6
        if bool(weak.any()):
            fallback = torch.zeros_like(raw)
            fallback[:, 0] = 1.0
            fallback = fallback - (fallback * t_plus).sum(-1, keepdim=True) * t_plus
            raw[weak] = fallback[weak]
        direction = unit(raw)
        t_star = unit(t_plus + self.target_offset * direction)

        observed_embeddings = self.prototype_embeddings[: self.prototype_count]
        hidden_embeddings = self.prototype_embeddings[self.prototype_count :]
        positive_proto = (t_plus @ observed_embeddings.T).topk(self.n_positive, dim=1).indices
        banned = torch.zeros(
            count,
            self.prototype_count,
            dtype=torch.bool,
        )
        banned.scatter_(1, positive_proto, True)
        utility_geometry = (
            (t_plus[:, None, :] - observed_embeddings[None, :, :]) * direction[:, None, :]
        ).sum(-1)
        useful_proto = self._topk_excluding(
            utility_geometry,
            banned,
            1,
        ).squeeze(1)
        banned.scatter_(1, useful_proto[:, None], True)
        unhelpful_proto = self._topk_excluding(
            -utility_geometry,
            banned,
            1,
        ).squeeze(1)

        positive_pairs = torch.stack(
            [
                self.action_id(positive_proto, 0),
                self.action_id(positive_proto, 1),
            ],
            dim=-1,
        )
        cells = {
            "useful_common": self.action_id(useful_proto, 0),
            "useful_rare": self.action_id(useful_proto, 1),
            "unhelpful_common": self.action_id(unhelpful_proto, 0),
            "unhelpful_rare": self.action_id(unhelpful_proto, 1),
        }
        useful_pair = torch.stack(
            [cells["useful_common"], cells["useful_rare"]],
            dim=1,
        )
        unhelpful_pair = torch.stack(
            [cells["unhelpful_common"], cells["unhelpful_rare"]],
            dim=1,
        )

        reward_matrix = torch.full(
            (count, self.action_count),
            self.neutral_reward,
        )
        reward_matrix.scatter_(
            1,
            positive_pairs.reshape(count, -1),
            torch.full(
                (count, self.n_positive * 2),
                self.positive_reward,
            ),
        )
        for cell in ("useful_common", "useful_rare"):
            reward_matrix.scatter_(
                1,
                cells[cell][:, None],
                torch.full((count, 1), self.useful_reward),
            )
        for cell in ("unhelpful_common", "unhelpful_rare"):
            reward_matrix.scatter_(
                1,
                cells[cell][:, None],
                torch.full((count, 1), self.unhelpful_reward),
            )

        hidden_similarity = t_star @ hidden_embeddings.T
        hidden_rewards = (
            self.hidden_reward_min
            + (self.hidden_reward_max - self.hidden_reward_min) * (hidden_similarity + 1.0) * 0.5
        )
        reward_matrix[:, self.observed_action_count :] = hidden_rewards
        hidden_optimal_actions = (
            hidden_rewards.topk(
                self.hidden_optimal_count,
                dim=1,
            ).indices
            + self.observed_action_count
        )
        return {
            "states": states,
            "t_plus": t_plus,
            "direction": direction,
            "t_star": t_star,
            "reward_matrix": reward_matrix,
            "hidden_optimal_actions": hidden_optimal_actions,
            "positive_proto": positive_proto,
            "positive_pairs": positive_pairs,
            "useful_proto": useful_proto,
            "unhelpful_proto": unhelpful_proto,
            "useful_pair": useful_pair,
            "unhelpful_pair": unhelpful_pair,
            **cells,
            **{
                f"{name}_advantage": torch.full(
                    (count,),
                    self.negative_advantage,
                )
                for name in CELL_NAMES
            },
        }

    def initial_rarity_half_gap(self) -> float:
        return self.protocol.initial_rarity_logit_gap / 2.0

    def audit(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seed": self.seed,
            "splits": {},
        }
        passed = True
        for name, split in (
            ("train", self.train),
            ("test", self.test),
        ):
            rows = torch.arange(len(split["states"]))
            useful_common = split["useful_common"]
            useful_rare = split["useful_rare"]
            unhelpful_common = split["unhelpful_common"]
            unhelpful_rare = split["unhelpful_rare"]
            advantages = torch.stack(
                [split[f"{cell}_advantage"] for cell in CELL_NAMES],
                dim=1,
            )
            checks = {
                "useful_replica_same_semantic_prototype": bool(
                    torch.equal(
                        useful_common // 2,
                        useful_rare // 2,
                    )
                ),
                "unhelpful_replica_same_semantic_prototype": bool(
                    torch.equal(
                        unhelpful_common // 2,
                        unhelpful_rare // 2,
                    )
                ),
                "rarity_replica_identity_exact": bool(
                    torch.all(useful_common % 2 == 0)
                    and torch.all(useful_rare % 2 == 1)
                    and torch.all(unhelpful_common % 2 == 0)
                    and torch.all(unhelpful_rare % 2 == 1)
                ),
                "negative_advantage_equal": bool(torch.all(advantages == self.negative_advantage)),
                "useful_reward_exact": bool(
                    torch.allclose(
                        split["reward_matrix"][
                            rows,
                            useful_common,
                        ],
                        torch.full(
                            (len(rows),),
                            self.useful_reward,
                        ),
                    )
                    and torch.allclose(
                        split["reward_matrix"][
                            rows,
                            useful_rare,
                        ],
                        torch.full(
                            (len(rows),),
                            self.useful_reward,
                        ),
                    )
                ),
                "unhelpful_reward_exact": bool(
                    torch.allclose(
                        split["reward_matrix"][
                            rows,
                            unhelpful_common,
                        ],
                        torch.full(
                            (len(rows),),
                            self.unhelpful_reward,
                        ),
                    )
                    and torch.allclose(
                        split["reward_matrix"][
                            rows,
                            unhelpful_rare,
                        ],
                        torch.full(
                            (len(rows),),
                            self.unhelpful_reward,
                        ),
                    )
                ),
                "hidden_ids_valid": bool(
                    torch.all(split["hidden_optimal_actions"] >= self.observed_action_count)
                    and torch.all(split["hidden_optimal_actions"] < self.action_count)
                ),
            }
            split_passed = all(checks.values())
            passed = passed and split_passed
            result["splits"][name] = {
                "passed": split_passed,
                **checks,
            }
        result.update(
            {
                "passed": passed,
                "protocol_revision": self.protocol.protocol_revision,
                "observed_action_count": self.observed_action_count,
                "hidden_action_count": self.hidden_action_count,
                "action_count": self.action_count,
                "hidden_actions_share_rare_coordinate": bool(
                    torch.all(self.action_rarity_sign[self.observed_action_count :] < 0)
                ),
                "trainable_per_action_bias": False,
            }
        )
        return result
