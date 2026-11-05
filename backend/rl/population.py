"""Population Based Training: how the agents of one generation are put in order.

Pure logic, no database and no network, so it can be tested on its own.

Every agent plays the same fixed champion. Its score is what it gets in that match
(win = 1, draw = 0.5, loss = 0, as a share of the games). The better score ranks higher.
When two agents have the same score, the one with the LOWER training loss ranks higher."""
from __future__ import annotations


def total_loss(policy_loss: float | None, value_loss: float | None) -> float | None:
    """One number for the tie-break: policy loss + value loss.
    None when the agent has no loss at all (it never trained)."""
    parts = [x for x in (policy_loss, value_loss) if x is not None]
    return float(sum(parts)) if parts else None


def rank_agents(results: list[dict]) -> list[dict]:
    """results: [{"key": anything, "score": float, "loss": float | None}, ...]

    Returns the same dicts, best first, each with a "rank" (1 = best). Ranks are always
    1..n without gaps, so the next phase can take "the first k" without special cases.

    Order: higher score first. Same score: lower loss first. An agent with no loss comes
    after every agent that has one (it has nothing to prove it is better). Same score and
    same loss: the order the agents came in, so the result never depends on chance."""
    order = sorted(range(len(results)), key=lambda i: (
        -results[i]["score"],
        results[i]["loss"] is None,
        results[i]["loss"] if results[i]["loss"] is not None else 0.0,
        i))
    return [{**results[i], "rank": pos} for pos, i in enumerate(order, start=1)]


def select(ranked: list[dict], survivors: int) -> dict:
    """Exploit: who stays and who is replaced after a generation.

    ranked: the agents of the generation, each with a "rank" (see rank_agents or the member
    rows saved by score_generation). Agents without a rank (their model was deleted, so
    they were never scored) are left out of the choice.

    survivors: how many stay (PopulationConfig.survivors). It is cut down to at least 1 and
    at most len - 1, so somebody always stays and somebody is always replaced.

    Returns {"survivors": [...], "replaced": [...], "pairs": [(replaced, parent), ...],
    "champion": best agent or None}. survivors and replaced are best first; every replaced
    agent gets a survivor as its parent (see parents_for)."""
    pool = sorted((a for a in ranked if a.get("rank") is not None), key=lambda a: a["rank"])
    if not pool:
        return {"survivors": [], "replaced": [], "pairs": [], "champion": None}
    if len(pool) == 1:                       # nobody to replace and nobody to copy from
        return {"survivors": pool, "replaced": [], "pairs": [], "champion": pool[0]}
    k = max(1, min(int(survivors), len(pool) - 1))
    stay, out = pool[:k], pool[k:]
    return {"survivors": stay, "replaced": out, "pairs": parents_for(out, stay),
            "champion": stay[0]}


def parents_for(replaced: list[dict], survivors: list[dict]) -> list[tuple[dict, dict]]:
    """Give every replaced agent one survivor to copy. Survivors are taken in turn, best
    first, so the shares are as equal as possible and, when they cannot be equal, the
    stronger survivors get the extra children. With 8 replaced and 2 survivors: 4 and 4."""
    if not survivors:
        return []
    return [(r, survivors[i % len(survivors)]) for i, r in enumerate(replaced)]


WHOLE_NUMBERS = {"simulations"}          # these stay whole numbers; every other value is a float


def mutate_hyperparams(hyperparams: dict, strength: float, limits: dict, rng) -> dict:
    """Explore: nudge the settings of a child a little.

    hyperparams: {"mcts": {"c_puct": 1.5, ...}, "train": {"lr": 0.001}} (what the parent
    trained with). Every number is multiplied, on its own, by 1 - strength or 1 + strength
    (0.8 or 1.2 with the default 0.2), the choice made by `rng` (a random.Random). The result
    is always cut to `limits` ({section: {name: (low, high)}}, the website's LIMITS), so a
    wrong number can never come out. simulations stays a whole number; the others are floats
    even when they were saved as 2 instead of 2.0. strength 0 gives an exact copy. The input is not changed."""
    out = {}
    for section, values in hyperparams.items():
        out[section] = {}
        for name, v in values.items():
            if strength:
                new = v * (1 + strength if rng.random() < 0.5 else 1 - strength)
                whole = name in WHOLE_NUMBERS
                if whole:
                    new = int(round(new))
                lo_hi = limits.get(section, {}).get(name)
                if lo_hi:
                    new = min(max(new, lo_hi[0]), lo_hi[1])
                v = int(new) if whole else float(new)
            out[section][name] = v
    return out
