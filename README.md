# EA Practical Assignment – Part I

Evolutionary Algorithms course, LIACS (Leiden University), 2026–2027.

We use a **Genetic Algorithm (GA)** to solve two problems from the IOHprofiler PBO suite:

- **F18 – LABS** (Low Autocorrelation Binary Sequences): 50 bits, maximize (best known ≈ 8.17)
- **F23 – N-Queens**: 49 bits = a 7×7 board, maximize (optimum = 7)

Each run may use at most **5,000 function evaluations**. The same hyperparameters are used for both problems.

## Algorithm

An **elitist GA** with:

- **Tournament selection**: pick `k` random individuals and choose the best one as a parent
- **Uniform crossover**: each bit of the child comes from parent 1 or parent 2 (coin flip)
- **Bit-flip mutation**: each bit is flipped with a small probability
- **Elitism**: the best individual always goes to the next generation unchanged

## Files

| File | What it does |
|---|---|
| `GA.py` | The genetic algorithm. Runs 20 independent runs on F18 and F23 |
| `tuning.py` | Hyperparameter tuning (max. 100,000 evaluations in total) |

## Requirements

```
pip install ioh numpy
```

## How to run

```
python GA.py
```

The results are saved in the `data/` folder. Zip the `run` folders and upload them to
[IOHanalyzer](https://iohanalyzer.liacs.nl/) to make plots.

## Hyperparameters

The GA function takes the hyperparameters as optional arguments:

```python
studentnumber1_studentnumber2_GA(problem, pop_size=20, tournament_size=3,
                                 crossover_rate=0.8, mutation_rate=0.02)
```

Calling it with only `problem` uses the default values.

## Reproducibility

The random seed is fixed (`np.random.seed(42)` in `GA.py`), so running the code
always gives the same results.

## Results (default settings, before tuning)

| Problem | Average best fitness (20 runs) |
|---|---|
| F18 (LABS) | 4.02 |
| F23 (N-Queens) | 6.05 (4 of 20 runs reached the optimum 7) |
