"""Hyper-parameters. Field names match the website's Settings page."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


class _Cfg:
    @classmethod
    def from_dict(cls, d: dict | None):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MCTSConfig(_Cfg):
    simulations: int = 100          # per move (website range 50-800)
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25     # 0 disables root noise
    temperature: float = 1.0
    temperature_moves: int = 20     # plies played with the temperature, then greedy
    leaves_per_batch: int = 8       # leaves of ONE game sent to the network together
    virtual_loss: float = 1.0
    reuse_tree: bool = True
    max_plies: int = 400            # longer games are adjudicated as draws


@dataclass
class TrainConfig(_Cfg):
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    games_per_iteration: int = 50
    epochs_per_iteration: int = 1   # passes over the replay buffer per iteration
    max_steps_per_iteration: int = 400
    buffer_size: int = 200_000
    min_buffer: int = 2_000         # no training before this many positions exist
    concurrent_games: int = 32      # games played at once per self-play process
    arena_games: int = 20
    arena_simulations: int = 100
    arena_threshold: float = 0.55   # score needed to replace the best model


@dataclass
class PopulationConfig(_Cfg):
    """Population Based Training: many agents of one model train side by side.

    After every generation the strongest agents stay as they are (exploit); the others are
    replaced by copies of them whose hyper-parameters are nudged a little (explore)."""
    size: int = 10                      # agents in the population (website range 2-100)
    survivor_fraction: float = 0.25     # share of the strongest agents that stay (0.05-0.5)
    iterations_per_generation: int = 3  # iterations every agent trains before they are ranked
    mutation_strength: float = 0.2      # a nudged value is multiplied by 1 - s or 1 + s; 0 = no explore
    copy_buffer: bool = True            # children also get a copy of their parent's replay buffer

    def __post_init__(self):
        # the settings check accepts any number, so 10.0 or 10.5 may arrive for a whole number
        self.size = int(self.size)
        self.iterations_per_generation = int(self.iterations_per_generation)
        self.copy_buffer = bool(self.copy_buffer)

    @property
    def survivors(self) -> int:
        """Agents that stay after a generation: at least 1, and always fewer than `size`."""
        n = int(self.size * self.survivor_fraction + 1e-9)   # + 1e-9: 0.29 * 100 must give 29
        return max(1, min(n, self.size - 1))

    @property
    def replaced(self) -> int:
        """Agents that are replaced by children of the survivors."""
        return self.size - self.survivors


# The settings that "explore" nudges when a child is made from a survivor (phase 7), as
# {section: names}. They are stored with every population member, so a child can start from
# the exact values its parent trained with.
EXPLORABLE = {
    "mcts": ("simulations", "c_puct", "temperature", "dirichlet_eps"),
    "train": ("lr",),
}


def explorable_values(settings: dict) -> dict:
    """{"mcts": {...}, "train": {...}}: just the explorable values of a full settings dict."""
    return {sec: {k: settings[sec][k] for k in keys} for sec, keys in EXPLORABLE.items()}
