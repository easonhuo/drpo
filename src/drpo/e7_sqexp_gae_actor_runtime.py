"""Public actor-runtime facade."""
from drpo.e7_sqexp_gae_actor_train import train_actor_branch
from drpo.e7_sqexp_gae_aggregate import aggregate_results

__all__ = ["train_actor_branch", "aggregate_results"]
