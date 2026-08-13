# CASCADE constraint structure and hybrid solver

## Information available to Eve

The attacker-facing solver consumes only:

- Eve's measurement value at each sifted position;
- the announced basis match, which marks that measurement as reliable or
  uninformative; and
- each public CASCADE query's key indices and Alice parity answer.

Alice's key and Bob's realized wrong-basis measurement outcomes are simulator
ground truth.  They are used after solving to score accuracy, never to construct
the problem.

## Algebraic classification

For a fixed pre-reconciliation Alice key `x`, every captured CASCADE answer is

```text
XOR(x[i] for i in block) = announced_parity.
```

These rows are affine equations over GF(2).  Duplicate and nested CASCADE
queries make many rows linearly dependent, so the number of independent leaked
bits is `rank(A)`, not the raw number of queries.

The measurement model is different:

- At correct-basis positions, Eve's value has error probability below 1/2.
  Maximum likelihood therefore minimizes the number of disagreements at these
  positions.  This is a cardinality objective, not an affine equation.
- At wrong-basis positions, Eve knows the position is wrong-basis but the value
  is uniform.  Marginalizing the unknown outcome produces no Boolean constraint
  on Alice's bit.
- A Hamming-distance interval against Bob's simulator-only key is a nonlinear
  pseudo-Boolean constraint.  The previous solver added two such constraints
  for the wrong-basis subset and the full key.  Both embed unavailable Bob bits.

## Hybrid algorithm

1. Express Alice bits as errors relative to Eve's known measurements.
2. Run bit-packed, vectorized Gauss-Jordan elimination with uninformative
   positions ordered first. This deduplicates the transcript, detects
   contradictions, computes its rank, and existentially eliminates the uniform
   positions.
3. The remaining projected rows are XOR equations only on reliable error bits.
4. Use the RREF particular solution as a valid cardinality upper bound.
5. Count independent unit rows that force reliable errors. When this lower
   bound equals the feasible upper bound, optimality is certified without SAT.
6. Otherwise, ask Z3 incrementally whether a lower-cardinality reliable error
   pattern exists. Stricter bounds retain learned implications between checks.
7. Back-substitute through the affine basis and verify every original public
   parity before returning.

This is not a claim that the complete problem is “just GF(2).”  Gaussian
elimination handles the affine core; SAT/pseudo-Boolean reasoning proves the
residual maximum-likelihood cardinality optimum.

Run correctness tests with:

```bash
PYTHONPATH=$PWD/cascade-python:$PWD python3 -m unittest -v test_hybrid_constraint_solver.py
```

Run the reproducible benchmark with:

```bash
PYTHONPATH=$PWD/cascade-python:$PWD python3 benchmark_hybrid_solver.py
```
