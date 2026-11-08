"""
Day 01 - Probability and Bayes' theorem
Application: estimating how reliable a supplier is, from delivery history.

Run:
    python3 main.py

Standard library only. No installs needed.
"""

import random
from dataclasses import dataclass


@dataclass
class BetaBelief:
    """
    Belief about an unknown probability p, stored as a Beta(alpha, beta) distribution.

    Bayes' rule in one line:
        posterior = prior x likelihood / evidence

    For a Beta prior and Bernoulli (on-time / late) observations, the update is just counting:
        alpha <- alpha + late_count
        beta  <- beta  + on_time_count
    """
    alpha: float = 1.0  # prior "pseudo-count" of late deliveries
    beta: float = 1.0   # prior "pseudo-count" of on-time deliveries

    def update(self, late: int, on_time: int) -> "BetaBelief":
        return BetaBelief(self.alpha + late, self.beta + on_time)

    @property
    def mean(self) -> float:
        """Expected probability of a late delivery."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def credible_interval(self, mass: float = 0.9, samples: int = 20_000, seed: int = 0):
        """Approximate central credible interval by Monte Carlo sampling."""
        rng = random.Random(seed)
        draws = sorted(rng.betavariate(self.alpha, self.beta) for _ in range(samples))
        lo_idx = int((1 - mass) / 2 * samples)
        hi_idx = int((1 + mass) / 2 * samples) - 1
        return draws[lo_idx], draws[hi_idx]


def simulate_deliveries(true_late_rate: float, n: int, seed: int = 42) -> list[bool]:
    """Simulate n deliveries. True means late. This stands in for real shipment logs."""
    rng = random.Random(seed)
    return [rng.random() < true_late_rate for _ in range(n)]


def summarize(belief: BetaBelief, label: str) -> str:
    lo, hi = belief.credible_interval()
    return (
        f"{label:<28} expected late rate = {belief.mean:6.2%}   "
        f"90% credible interval = [{lo:6.2%}, {hi:6.2%}]"
    )


def main() -> None:
    supplier_true_late_rate = 0.18  # hidden truth we are trying to learn

    # Step 1: prior. We are fairly unsure. Mean 50%, but weak (equivalent to 2 observations).
    prior = BetaBelief(alpha=1, beta=1)
    print(summarize(prior, "Prior (no data yet)"))

    # Step 2: observe deliveries in batches and update the belief each time.
    history = simulate_deliveries(supplier_true_late_rate, n=120)
    belief = prior
    for batch_start in range(0, len(history), 30):
        batch = history[batch_start : batch_start + 30]
        late = sum(batch)
        belief = belief.update(late=late, on_time=len(batch) - late)
        label = f"After {batch_start + len(batch):>3} deliveries"
        print(summarize(belief, label))

    print()
    print(f"Hidden true late rate:        {supplier_true_late_rate:6.2%}")
    print(f"Learned late rate (posterior): {belief.mean:6.2%}")

    # Step 3: a decision. Should we dual-source this part? Use the posterior, not a single number.
    threshold = 0.20
    p_above = 1 - _beta_cdf(belief, threshold)
    print()
    print(f"P(late rate > {threshold:.0%}) = {p_above:.1%}")
    if p_above > 0.5:
        print("Decision: the supplier is more likely than not to exceed the threshold. Add a backup source.")
    else:
        print("Decision: evidence does not yet support switching. Keep monitoring.")


def _beta_cdf(belief: BetaBelief, x: float, samples: int = 50_000, seed: int = 1) -> float:
    """Monte Carlo estimate of P(p <= x) under the belief."""
    rng = random.Random(seed)
    hits = sum(rng.betavariate(belief.alpha, belief.beta) <= x for _ in range(samples))
    return hits / samples


if __name__ == "__main__":
    main()
