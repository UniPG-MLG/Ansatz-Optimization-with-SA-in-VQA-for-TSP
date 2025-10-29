import random
from src.Individual import Individual
import math
import os
import copy
from src.logger import setup_logger, get_logger_files
from src.args import args
import src.myflow as myflow
from src.uuid import RUN_UUID
import signal

logger = setup_logger()

# Flag to control the loop
_running = True
_SAVE_DIR = "Best_ansatz"


def _simulated_annealing(
    eval_file: str = "",
    initial_temp: float = 1.0,
    cooling_rate: float = 0.9,
    min_temp: float = 1e-3,
    max_iterations: int = 10,
) -> Individual:
    """
    Perform the Simulated Annealing optimization algorithm to find the best solution.
    Simulated Annealing is a probabilistic technique for approximating the global optimum
    of a given function. This implementation starts with an initial solution and iteratively
    explores neighboring solutions, accepting worse solutions with a probability that decreases
    as the temperature decreases.

    Args:
        eval_file (str): The path to the evaluation file used to initialize the solution.
        initial_temp (float, optional): The initial temperature for the annealing process.
            Defaults to 1.0.
        cooling_rate (float, optional): The rate at which the temperature decreases.
            Defaults to 0.9.
        min_temp (float, optional): The minimum temperature at which the algorithm stops.
            Defaults to 1e-3.
        max_iterations (int, optional): The maximum number of iterations to perform.
            Defaults to 5000.

    Returns:
        Individual: The best solution found during the optimization process.
    """

    logger.info("Starting Simulated Annealing")

    # Register the signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    result_file = f"results/sa_{RUN_UUID}.csv"
    results = []

    best = Individual(eval_file)
    current = copy.deepcopy(best)
    T = initial_temp
    iteration = 0

    myflow.connect(
        run_name=args.run_name,
        nolog=args.nomlflow,
        log_system_metrics=args.log_system_metrics,
    )
    myflow.log_params(vars(args))
    myflow.log_params({"UUID": RUN_UUID})

    # Main loop: continue until temperature is below the minimum
    #            or max iterations are reached
    while _running and (T > min_temp and iteration < max_iterations):
        # Generate a neighbor solution by mutating the current solution
        neighbor = copy.deepcopy(current)
        neighbor.mutate()

        # Calculate the change in fitness between neighbor and current solution
        delta_e = neighbor.fitness - current.fitness

        myflow.log_metric("current_fitness", current.fitness, step=iteration)
        myflow.log_metric("neighbor_fitness", neighbor.fitness, step=iteration)
        myflow.log_metric("delta_e", delta_e, step=iteration)

        # Accept the neighbor solution based on the Metropolis criterion
        if delta_e > 0 or random.random() < math.exp(delta_e / T):
            current = neighbor
            if current.fitness > best.fitness:
                best = current
                best.save()

        # Decrease the temperature according to the cooling rate
        T *= cooling_rate

        myflow.log_metric("best_fitness", best.fitness, step=iteration)
        myflow.log_metric("T", T, step=iteration)

        results.append(
            (
                iteration,
                current.fitness,
                best.fitness,
                T,
            )
        )

        logger.info(
            "   Iteration %s: Current fitness: %s, Best fitness: %s, Temperature: %s",
            iteration,
            current.fitness,
            best.fitness,
            T,
        )

        iteration += 1

    _best_fitness = best.fitness

    if args.vqa_optimize_once:
        # force args False in order to enable optimizers
        args.vqa_optimize_once = False
        _recomputed = best._Individual__eval()  # force fitness update
        _best_fitness = _recomputed

        myflow.log_metric("best_fitness", _recomputed, step=iteration)

    logger.info(
        "Best ansatz found: \n fitness: %s\n%s",
        _best_fitness,
        str(best),
    )

    save_circuit(best, _SAVE_DIR, f"best_ansatz_{RUN_UUID}")

    # save results to file
    # TODO: should we save results while algo is running ??
    logger.info("Saving results to file %s", result_file)
    with open(result_file, "w") as f:
        f.write("iteration,current_fitness,best_fitness,T\n")
        for line in results:
            f.write(",".join((str(x) for x in line)))
            f.write("\n")

    # mlflow log artifacts
    for file in [
        result_file,
        f"results/results_vqa_tsp_{RUN_UUID}.csv",
        f"Best_ansatz/best_ansatz_{RUN_UUID}.png",
        f"results/best_ansatz_with_params_{RUN_UUID}.qpy",
    ]:
        myflow.log_artifact(file, artifact_name="results")

    for logger_file in get_logger_files():
        myflow.log_artifact(logger_file, artifact_name="logs")

    return best


def save_circuit(individual: Individual, directory: str, name: str) -> None:
    """
    Save the circuit diagram of the individual to a specified directory.

    Args:
        individual (Individual): The individual whose circuit diagram is to be saved.
        directory (str): The directory where the circuit diagram will be saved.
        name (str): The name of the file to save the circuit diagram as.

    returns:
        None
    """

    logger.info("Saving best ansatz to file")

    os.makedirs(directory, exist_ok=True)
    qc = individual.qc.decompose()

    try:
        fig = qc.draw(output="mpl")
        png_path = os.path.join(directory, f"{name}.png")
        fig.savefig(png_path)
    except Exception as e:
        logger.warning(
            "Warning: could not save visual diagram for %s: %s", name, e)


def handle_sigint(signum, frame):
    global _running
    logger.warning("SIGINT received. Exiting gracefully...")
    _running = False
