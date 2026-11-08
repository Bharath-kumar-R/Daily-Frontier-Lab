"""Tests for Day 01. Run with: python3 -m unittest test_main -v"""

import unittest

from main import BetaBelief, simulate_deliveries


class TestBetaBelief(unittest.TestCase):
    def test_uniform_prior_mean_is_half(self):
        self.assertAlmostEqual(BetaBelief(1, 1).mean, 0.5)

    def test_update_counts_observations(self):
        updated = BetaBelief(1, 1).update(late=3, on_time=7)
        self.assertEqual((updated.alpha, updated.beta), (4, 8))

    def test_posterior_mean_moves_toward_data(self):
        belief = BetaBelief(1, 1).update(late=10, on_time=0)
        self.assertGreater(belief.mean, 0.5)

    def test_more_data_shrinks_variance(self):
        small = BetaBelief(1, 1).update(late=2, on_time=8)
        large = small.update(late=20, on_time=80)
        self.assertLess(large.variance, small.variance)

    def test_update_is_order_independent(self):
        a = BetaBelief(1, 1).update(2, 5).update(4, 1)
        b = BetaBelief(1, 1).update(4, 1).update(2, 5)
        self.assertEqual((a.alpha, a.beta), (b.alpha, b.beta))

    def test_credible_interval_contains_mean(self):
        belief = BetaBelief(1, 1).update(late=18, on_time=82)
        lo, hi = belief.credible_interval(samples=5_000)
        self.assertLess(lo, belief.mean)
        self.assertGreater(hi, belief.mean)


class TestSimulation(unittest.TestCase):
    def test_simulation_is_reproducible(self):
        self.assertEqual(simulate_deliveries(0.3, 50, seed=7), simulate_deliveries(0.3, 50, seed=7))

    def test_simulated_rate_is_close_to_truth(self):
        draws = simulate_deliveries(0.25, 20_000, seed=3)
        self.assertAlmostEqual(sum(draws) / len(draws), 0.25, delta=0.02)


if __name__ == "__main__":
    unittest.main()
