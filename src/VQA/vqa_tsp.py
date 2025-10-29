import numpy as np
from .tsp_problem import TSP_problem
from scipy.optimize import minimize
import os
import glob
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2
from qiskit_aer import AerSimulator
from src.logger import setup_logger
from src.args import args
from src.uuid import RUN_UUID

logger = setup_logger()


class VQA_TSP:
    def __init__(self, filename: str, ansatz: QuantumCircuit) -> None:
        self.problem = TSP_problem(filename)
        self.n_cities = len(self.problem.distance_matrix)
        backend = AerSimulator(method="statevector", device="CPU")
        self.simulator = SamplerV2.from_backend(backend)
        self.ansatz = ansatz

    def _create_ansatz(self, reps: int) -> None:
        """
        Creates the ansatz for the VQA algorithm.
        The ansatz is a quantum circuit that will be used to represent the solution space.
        The number of qubits is determined by the number of cities in the TSP instance.
        The number of parameters is determined by the factorial of the number of cities.
        """

        fact = np.prod(range(1, self.n_cities + 1))
        n_qubits = int(0.99 + np.log2(fact))
        self.n_parameter = self.ansatz.num_parameters
        logger.debug(
            "Created ansatz with %s qubits and %s parameters",
            n_qubits,
            self.n_parameter,
        )
        self.ansatz.measure_all()

    def simulate(
        self,
        params: np.ndarray,
        n_shots: int = 1024,
        testing=False,
    ) -> dict[int, str]:
        """
        Simulates the quantum circuit with the provided parameters.

        Args:
            params (np.ndarray): Parameters for the ansatz circuit.
            n_shots (int): Number of shots for the simulation.

        Returns:
            dict[int, str]: Dictionary of counts representing the tours.
        """

        _params = params
        circuit = self.ansatz

        job = self.simulator.run([(circuit.decompose(), _params, n_shots)])
        res = job.result()[0]
        counts = res.data.meas.get_counts()
        return counts

    def _objective_function_avg(self, params: np.ndarray, n_shots: int = 1024) -> float:
        """
        Objective function to be minimized.
        This function calculates the average cost of the tour represented by the quantum circuit.
        The cost is calculated by evaluating the circuit with the given parameters and
        counting the frequency of each tour.
        The average cost is then returned.

        Args:
            params (np.ndarray): Parameters for the ansatz circuit.
            n_shots (int): Number of shots for the simulation.

        Returns:
            float: Average cost of the tour.
        """

        counts = self.simulate(params, n_shots)
        total = sum(
            freq * self.problem.evaluateb(bs[::-1]) for bs, freq in counts.items()
        )

        return total / n_shots

    def run(
        self,
        num_tries: int = 10,
        max_iter_spa: int = 1000,
        max_iter_cobyla: int = 1000,
        instance_name: str = "",
        run_id: int = 0,
        initial_params: np.ndarray | None = None,
        sampling: bool = True,
        testing=False,
    ):
        """
        Run the VQA algorithm for the TSP instance.
        This function performs the following steps:
        1. Finds the best solution using the problem's method.
        2. Generates random samples of parameters and evaluates their average cost.
        3. Uses the SPSA optimizer to minimize the average cost.
        4. Uses the COBYLA optimizer to further minimize the average cost.
        5. Simulates the final circuit and calculates the probability of finding the optimal solution.
        6. Logs the results.

        Args:
            num_tries (int): Number of random samples to generate.
            max_iter_spa (int): Maximum iterations for the SPSA optimizer.
            max_iter_cobyla (int): Maximum iterations for the COBYLA optimizer.
            instance_name (str): Name of the TSP instance.
            run_id (int): Run identifier for logging purposes.

        Returns:
            list: List of dictionaries containing the results of the run.
        """

        fmin, best_permutation = self.problem._find_best_solution()
        logger.debug(
            "Best solution found: %s with permutation %s",
            fmin,
            best_permutation,
        )

        # Sampling
        samples = []

        if sampling:
            for _ in range(num_tries**2):
                x_rand = 2 * np.pi * np.random.random(self.n_parameter)
                samples.append((x_rand, self._objective_function_avg(x_rand)))
            samples.sort(key=lambda c: c[1])
        else:
            samples = [
                (initial_params, self._objective_function_avg(initial_params))]
            num_tries = 1

        log_results = []

        for r in range(num_tries):
            x0 = samples[r][0]
            init_fun = samples[r][1]

            res_optim = None
            run_optimizer = (not args.vqa_optimize_once) or (not sampling)

            if run_optimizer:
                if "spsa" in args.optimizers:
                    res_spsa = minimize.SPSA(
                        maxiter=max_iter_spa,
                        # smaller value to avoid division by zero
                        learning_rate=0.01,
                        # smaller value to reduce likelihood of division by zero
                        perturbation=0.001,
                        # increase this value for more accurate gradient estimates
                        resamplings=10,
                    ).minimize(
                        fun=self._objective_function_avg,
                        x0=x0,
                    )

                    res_optim = res_spsa

                # COBYLA
                if "cobyla" in args.optimizers:
                    res_cobyla = minimize(
                        fun=self._objective_function_avg,
                        x0=x0,
                        method="COBYLA",
                        options={"maxiter": max_iter_cobyla},
                    )

                    res_optim = res_cobyla

                # POWELL (gradient free)
                if "powell" in args.optimizers:
                    res_powell = minimize(
                        fun=self._objective_function_avg,
                        x0=x0,
                        method="Powell",
                        options={"maxiter": args.vqa_powell_iter},
                    )

                    res_optim = res_powell

                # BFGS (gradient based)
                if "bfgs" in args.optimizers:
                    res_bfgs = minimize(
                        fun=self._objective_function_avg,
                        x0=x0,
                        method="BFGS",
                        options={"maxiter": args.vqa_bfgs_iter},
                    )

                    res_optim = res_bfgs

            _params = res_optim.x if (
                run_optimizer and res_optim is not None) else x0

            counts = self.simulate(
                _params,
                n_shots=1024,
                testing=testing,
            )
            nshots = 1024

            num_success = sum(
                freq
                for bs, freq in counts.items()
                if self.problem.evaluateb(bs[::-1]) == fmin
            )
            prob_opt = num_success / nshots

            # print warning when prob_opt too low
            if prob_opt <= 0.02:
                logger.warning(
                    "PROB OPT TOO LOW. ANSATZ: %s",
                    prob_opt,
                )

            log_results.append(
                {
                    "instance": instance_name,
                    "run": run_id,
                    "n_cities": self.problem.n_cities,
                    "n_tries": r + 1,
                    "e_min": fmin,
                    "initial_average": init_fun,
                    "final_average": res_optim.fun
                    if (run_optimizer and res_optim is not None)
                    else 0.0,
                    "prob_opt": prob_opt,
                    "params": _params,
                }
            )
        return log_results


def run_experiments(
    ansatz: QuantumCircuit,
    instance_pattern: str = "tsp_instance_*.json",
    runs_per_instance: int = 3,
    spa_iter: int = 1000,
    cobyla_iter: int = 1000,
    initial_params: np.ndarray | None = None,
    sampling: bool = True,
    testing=False,
):
    """
    Run experiments for the VQA TSP algorithm.
    This function processes all instances matching the given pattern,
    runs the VQA algorithm for each instance, and saves the results to a CSV file.

    Args:
        instance_pattern (str): Pattern to match instance files.
        runs_per_instance (int): Number of runs for each instance.

    Returns:
        list: List of all logs from the experiments.
    """
    all_logs = []
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"results_vqa_tsp_{RUN_UUID}.csv")

    instance_files = sorted(glob.glob(instance_pattern))
    for instance_file in instance_files:
        instance_name = os.path.basename(instance_file)
        logger.debug(
            "Processing instance: %s",
            instance_name,
        )
        vqa = VQA_TSP(instance_file, ansatz)
        vqa._create_ansatz(reps=1)

        for run_id in range(1, runs_per_instance + 1):
            logger.debug(
                "Running instance: %s, Run: %s",
                instance_name,
                run_id,
            )
            logs = vqa.run(
                num_tries=args.vqa_num_tries,
                max_iter_spa=spa_iter,
                max_iter_cobyla=cobyla_iter,
                instance_name=instance_name,
                initial_params=initial_params,
                sampling=sampling,
                run_id=run_id,
                testing=testing,
            )
            all_logs.extend(logs)
            df = pd.DataFrame(all_logs)
            df.to_csv(csv_path, index=False)
            logger.debug(
                "Saved results to %s",
                csv_path,
            )

    logger.debug("All experiments completed.")
    return all_logs


def needed_qubits(instance_file: str):
    problem = TSP_problem(instance_file)
    n_cities = len(problem.distance_matrix)

    fact = np.prod(range(1, n_cities + 1))
    n_qubits = int(0.99 + np.log2(fact))

    return n_qubits
