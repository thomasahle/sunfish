# Tuning tools

The two tuners answer different questions:

- [`texel/`](texel/) fits static evaluation tables from labelled positions.
- [`logistic_gp/`](logistic_gp/) optimizes arbitrary UCI parameters from noisy
  game outcomes, while keeping a deliberate exploration budget.

The game-result tuner is engine-independent. Its included parameter space and
correctness gate are Sunfish examples, not requirements of the optimizer.
