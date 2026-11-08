# Day 01: Probability and Bayes' Theorem

**Industry lens:** supply chain. Estimating how often a supplier delivers late, and deciding when to add a backup source.

## 1. Foundation (history)

In 1763, Thomas Bayes' essay, published after his death, showed how to update a belief when new evidence arrives. Pierre-Simon Laplace later developed the idea into the general rule used today. For about two centuries it was overshadowed by frequentist statistics, then came back as computers made the calculations practical.

The core idea is simple:

> **posterior = prior × likelihood / evidence**

Start with a belief, look at the data, and revise the belief. Everything in modern machine learning, from spam filters to large language model training, is built on this same logic of updating beliefs from evidence.

## 2. Present

Bayesian updating is used in:

- Spam filters and medical test interpretation
- A/B testing and demand forecasting
- Probabilistic programming tools
- Bayesian optimization, which tunes many machine learning models

## 3. Future

- **Uncertainty-aware decisions:** systems that report how confident they are, not just a single answer.
- **Hybrid models:** combining prior domain knowledge (like supplier contracts) with learned patterns from data.
- **Open question:** how to keep priors honest when data is scarce or biased.

## 4. Build: what the code does

`main.py` simulates 120 deliveries from a supplier with a hidden 18% late rate, then:

1. Starts with a weak prior (Beta(1, 1), meaning "no idea").
2. Updates the belief in batches of 30 deliveries.
3. Reports the expected late rate and a 90% credible interval after each batch.
4. Turns the posterior into a decision: is the probability of exceeding a 20% late rate above 50%?

The **Beta distribution** is the key tool. With a Beta prior and on-time/late observations, the update is just counting: add late deliveries to alpha, and on-time deliveries to beta.

## Run it

```bash
cd days/2026-11-07-day-01-probability-and-bayes
python3 main.py
python3 -m unittest test_main -v
```

No dependencies. Python 3.9 or newer.

## Try next

- Replace the simulated data with a real shipment log CSV.
- Compare two suppliers and compute the probability that one is worse than the other.
- Change the prior and see how it affects the result after 10 deliveries versus 1,000.
