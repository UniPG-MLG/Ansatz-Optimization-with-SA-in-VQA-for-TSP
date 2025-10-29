import argparse

parser = argparse.ArgumentParser(
    "QLM - Simulated Annealing",
    formatter_class=argparse.RawTextHelpFormatter,
)

group_mlflow = parser.add_argument_group("MLFLOW", "Mlflow settings")

parser.add_argument(
    "--verbose",
    action="store_true",
    help="Be verbose (debug logging)",
)

## MLFLOW ARGS ##
group_mlflow.add_argument(
    "--nomlflow",
    action="store_true",
    help="Do not log statistics with mlflow",
)
group_mlflow.add_argument(
    "--log_system_metrics",
    action="store_true",
    help="Log cpu stats while running test. E.g.: CPU usage, RAM usage, ...",
)
group_mlflow.add_argument(
    "--run_name",
    default=None,
    type=str,
    help="Set mlflow run name",
)
group_mlflow.add_argument(
    "--uri",
    default="http://localhost:5000",
    type=str,
    help="Mlflow server's URI",
)

## Simulated Annealing ARGS ##
group_sa = parser.add_argument_group(
    "SA",
    "Simulated Annealing Algorithm settings",
)
group_sa.add_argument(
    "--max_iterations",
    default=100,
    type=int,
    help="Simulated Annealing max iterations",
)
group_sa.add_argument(
    "--initial_temp",
    default=1.0,
    type=float,
    help="Simulated Annealing initial temperature (T) value",
)
group_sa.add_argument(
    "--cooling_rate",
    default=0.9,
    type=float,
    help="Simulated Annealing cooling rate value",
)
group_sa.add_argument(
    "--eval_file",
    default="Instances/5_cities/tsp_instance_0.json",
    type=str,
    help="TSP instance used to test Ansatz and retrieve fitness values",
)
group_sa.add_argument(
    "--test_dir",
    default=None,
    type=str,
    help="TSP instances' dir used to test pre-optimized Ansatz after SA algorithm.",
)
group_sa.add_argument(
    "--qc_params",
    default="all",
    type=str,
    choices=["all", "one_per_block"],
    help="""Set the rotation block parameters:
        - all           : use one argument per Gate
        - onge_per_block: use just one argument per Rotation block
        """,
)

## VQA ARGS ##
group_VQA = parser.add_argument_group("VQA", "VQA Algorithm settings")
group_VQA.add_argument(
    "--vqa_optimize_once",
    action="store_true",
    help="""Run the optimization phase just onec at the end of Simulated Annealing.
    This means that every fitness is computed without the optimization phase and when
    the best ansatz is found, will be optimized (just onece) !
    """,
)
group_VQA.add_argument(
    "--optimizers",
    default=["cobyla"],
    choices=["cobyla", "powell", "bfgs", "spsa"],
    type=str,
    nargs="+",
    help="List of optimizers to use while running the QC.",
)
group_VQA.add_argument(
    "--vqa_runs_per_instance",
    default=3,
    type=int,
    help="For each instance run n runs_per_instance independent experiments.",
)
group_VQA.add_argument(
    "--vqa_num_tries",
    default=10,
    type=int,
    help="Number of starting points to be optimized in each run.",
)
group_VQA.add_argument(
    "--vqa_spsa_iter",
    default=1000,
    type=int,
    help="Number of iterations for SPA optimizer.",
)
group_VQA.add_argument(
    "--vqa_cobyla_iter",
    default=1000,
    type=int,
    help="Number of iterations for COBYLA optimizer.",
)
group_VQA.add_argument(
    "--vqa_powell_iter",
    default=1000,
    type=int,
    help="Number of iterations for Powell optimizer",
)
group_VQA.add_argument(
    "--vqa_bfgs_iter",
    default=1000,
    type=int,
    help="Number of iterations for BFGS optimizer",
)

args = parser.parse_args()
