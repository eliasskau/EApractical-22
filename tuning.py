from typing import List

import numpy as np
from GA import studentnumber1_studentnumber2_GA
from ioh import get_problem, ProblemClass
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/

budget = 100_000

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

# Hyperparameters to tune, e.g.
hyperparameter_space = {
    "population_size": [50, 100, 200],
    "mutation_rate": [0.01, 0.05, 0.1],
    "crossover_rate": [0.5, 0.7, 0.9],
}

def _rank_descending(values: List[float]) -> np.ndarray:
    # rank so higher value gets lower rank (rank 0 = best) since both problems are maximization
    order = np.argsort(values)[::-1]
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(len(values))
    return ranks


def _rank_sum_score(results: List[dict]) -> List[int]:
    labs_vals = [r["labs"] for r in results]
    nq_vals   = [r["nq"]   for r in results]
    return [
        int(l + n)
        for l, n in zip(_rank_descending(labs_vals),
                        _rank_descending(nq_vals))
    ]

# Hyperparameter tuning function
def tune_hyperparameters() -> List:
    best_params = None

    # create problems without logger during tuning to avoid c++ teardown crash
    F18 = get_problem(18, dimension=50, instance=1, problem_class=ProblemClass.PBO)
    F23 = get_problem(23, dimension=49, instance=1, problem_class=ProblemClass.PBO)

    # tuning values
    STAGE1_EVALS = 1000
    STAGE2_EVALS = 5000
    USED = 0
    results = []

    for pop_size in hyperparameter_space["population_size"]:
        for mutation_rate in hyperparameter_space["mutation_rate"]:
            for crossover_rate in hyperparameter_space["crossover_rate"]:
                # run ga on f18
                F18.reset()
                np.random.seed(42)
                f_labs = studentnumber1_studentnumber2_GA(
                    F18,
                    pop_size=pop_size,
                    tournament_size=3,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    max_evals=STAGE1_EVALS,
                )

                # run ga on f23
                F23.reset()
                np.random.seed(42)
                f_nq = studentnumber1_studentnumber2_GA(
                    F23,
                    pop_size=pop_size,
                    tournament_size=3,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    max_evals=STAGE1_EVALS,
                )

                # record
                USED += 2 * STAGE1_EVALS
                results.append({
                    "pop": pop_size,
                    "mr": mutation_rate,
                    "cr": crossover_rate,
                    "labs": f_labs,
                    "nq": f_nq,
                })
                print(f"[{len(results):2d}/27] pop={pop_size:3d} mr={mutation_rate:.3f} "
                      f"cr={crossover_rate:.1f} | labs={f_labs:.4f} nq={f_nq:.1f} | used={USED}")

                assert USED <= budget, f"Budget exceeded: {USED} > {budget}"

    # rank configs and pick survivors
    scores = _rank_sum_score(results)
    order  = np.argsort(scores)
    TOP_K  = 4
    survivors = [results[i] for i in order[:TOP_K]]

    print(f"\n=== Stage 1 complete: {USED} evals used ===")
    print(f"=== Top {TOP_K} survivors (rank-sum, lower = better): ===")
    for i in order[:TOP_K]:
        r = results[i]
        print(f"  rank_sum={scores[i]:2d} | pop={r['pop']:3d} mr={r['mr']:.3f} "
              f"cr={r['cr']:.1f} | labs={r['labs']:.4f} nq={r['nq']:.1f}")

    # stage 2 re-evaluate survivors at final budget
    stage2_results = []
    print(f"\n=== Stage 2: {len(survivors)} survivors @ {STAGE2_EVALS} evals ===")
    for i, cfg in enumerate(survivors):
        pop_size       = cfg["pop"]
        mutation_rate  = cfg["mr"]
        crossover_rate = cfg["cr"]

        # run ga on f18
        F18.reset()
        np.random.seed(42)
        f_labs = studentnumber1_studentnumber2_GA(
            F18,
            pop_size=pop_size,
            tournament_size=3,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_evals=STAGE2_EVALS,
        )

        # run ga on f23
        F23.reset()
        np.random.seed(42)
        f_nq = studentnumber1_studentnumber2_GA(
            F23,
            pop_size=pop_size,
            tournament_size=3,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_evals=STAGE2_EVALS,
        )

        # record
        USED += 2 * STAGE2_EVALS
        stage2_results.append({
            "pop": pop_size,
            "mr":  mutation_rate,
            "cr":  crossover_rate,
            "labs": f_labs,
            "nq": f_nq,
        })
        print(f"[{i+1}/{len(survivors)}] pop={pop_size:3d} mr={mutation_rate:.3f} "
              f"cr={crossover_rate:.1f} | labs={f_labs:.4f} nq={f_nq:.1f} | used={USED}")

        assert USED <= budget, f"Budget exceeded: {USED} > {budget}"

    # rank stage 2 results and pick winner
    stage2_scores = _rank_sum_score(stage2_results)
    stage2_order  = np.argsort(stage2_scores)

    print(f"\n=== Stage 2 complete: {USED} evals total ===")
    print(f"=== Final ranking (rank-sum, lower = better): ===")
    for i in stage2_order:
        r = stage2_results[i]
        print(f"  rank_sum={stage2_scores[i]:2d} | pop={r['pop']:3d} mr={r['mr']:.3f} "
              f"cr={r['cr']:.1f} | labs={r['labs']:.4f} nq={r['nq']:.1f}")

    winner = stage2_results[int(stage2_order[0])]
    best_params = [winner["pop"], winner["mr"], winner["cr"]]

    print(f"\n=== winner: pop={winner['pop']} mr={winner['mr']} cr={winner['cr']} ===")
    print(f"    labs = {winner['labs']:.4f}")
    print(f"    nq   = {winner['nq']:.1f}")
    print(f"    total tuning evals used: {USED} / {budget}")

    return best_params


if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)