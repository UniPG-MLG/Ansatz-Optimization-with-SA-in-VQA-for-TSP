import random
import numpy as np
from src.logger import setup_logger

from src.VQA import run_experiments, needed_qubits
from src.args import args
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import ParameterVector, QuantumCircuit, Parameter
from qiskit import qpy
from typing import Literal
from src.uuid import RUN_UUID


logger = setup_logger()


class Individual:
    GENES_MAPPING = {
        "R": {
            0: "rx",
            1: "ry",
            2: "rz",
        },
        "CNOT": {
            3: "linear",
            4: "reverse_linear",
            5: "full",
            6: "circular",
            7: "sca",
        },
    }

    GENE_SIZE = 5

    def __init__(self, eval_file: str = "Instances/6_cities/tsp_instance_0.json"):
        """Initialize an Individual

        An individual is made by 5 genes defined as follows:
            type:  R  CX  R  CX  R
            idx : [0,  1, 2,  3, 4]

        When a new individual is instantiated the genes are randomly created.
        The genes structure is fixed and cannot be altered (e.g. cannot exists 2 consecutive R genes)
        In a shorter form: genes with index odd are Entanglements (CNOT) and genes with even
        index are Rotations (Rx, Ry, Rz).

        When the relative Quantum Circuit is needed its computed on-the-fly by generating 5 TwoLocals circuits
        and then combining them together (dealing with vars names.)
        """
        self.eval_file = eval_file
        self.qubits = needed_qubits(self.eval_file)
        self.__fitness = None
        self.best_params = None
        # used to re-compute the fitness only when a mutation occurs
        self.__mutated = False
        logger.debug("QUBITS: %s | FILE: %s", self.qubits, self.eval_file)

        self.genes = self.__genes_init()

    @property
    def qc(self) -> QuantumCircuit:
        """Quantum Circuit associated with genes

        Its computed on-the-fly when needed.
        """
        qc = self.__genes_to_qc()
        return qc

    @property
    def fitness(self) -> float:
        """Return Individual fitness

        Fitness is computed only when needed:
            - When the genes are initialized
            - When a mutation occurs

        Returns:
            - (float) Fitness value
        """
        if self.__fitness is None:
            logger.debug("Compute fitness because: Never Computed")
            self.__fitness = self.__eval()

        elif self.__mutated:
            logger.debug("Compute fitness because: Mutation Occured")
            self.__fitness = self.__eval()
            self.__mutated = False

        return self.__fitness

    def save(self):
        """Save Ansatz to binary file
        Save Ansatz with best parameters found during various SA iterations
        into a binary file. This let us to use circuits with optimized
        paramas with other problems/scripts.
        """
        logger.debug(
            "Saving best ansatz (%s) with params: %s",
            self.best_params["prob_opt"],
            self.best_params["params"],
        )

        np.save(
            f"results/best_ansatz_with_params_{RUN_UUID}_params.npy",
            self.best_params["params"],
        )

    def __eval(self) -> float:
        """Evaluate the Individual (compute fitness)

        Computes the Individual's fitness by running the VQA algorithm several times
        and computing an AVG on the results

        Returns:
            - (float) Fitness value
        """
        results = run_experiments(
            instance_pattern=self.eval_file,
            ansatz=self.qc,
            runs_per_instance=args.vqa_runs_per_instance,
            spa_iter=args.vqa_spsa_iter,
            cobyla_iter=args.vqa_cobyla_iter,
        )

        avg_res = sum([x["prob_opt"] for x in results]) / len(results)

        # sort results by "prob_opt" in reverse order (best, second, third, ...)
        # and takes the best parameters found
        self.best_params = sorted(
            results,
            key=lambda x: x["prob_opt"],
            reverse=True,
        )[0]  # ["params"]

        return avg_res

    def mutate(self):
        """Mutation operation

        Performs the mutation operation as follows:
            - Randomly select a gene
            - Choose a new gene (following the gene rules) different from the selected one
            - Replace selected gene with new gene
        """
        logger.debug("MUTATION")

        idx_gene_to_mutate = random.randrange(Individual.GENE_SIZE)
        gene_type = self.__get_gene_type(idx_gene_to_mutate)
        actual_gene = self.genes[idx_gene_to_mutate]
        logger.debug("   SELECTED GENE: %s (idx: %s)",
                     actual_gene, idx_gene_to_mutate)

        possible_choices = list(Individual.GENES_MAPPING[gene_type].keys())
        possible_choices.remove(actual_gene)
        logger.debug("   NEW GENES: %s", possible_choices)

        new_gene = random.choice(possible_choices)

        logger.debug("   %s --> %s", actual_gene, new_gene)
        self.genes[idx_gene_to_mutate] = new_gene
        self.__mutated = True

    def __get_gene_type(self, gene: int) -> Literal["CNOT"] | Literal["R"]:
        """Given the gene index returns gene type

        Args:
            - gene (int): gene index

        Returns:
            - Gene type. It could be "CNOT" (if gene index is odd) or "R" (if gene index is even)
        """
        return "CNOT" if self.__is_odd(gene) else "R"

    def __is_odd(self, val: int) -> bool:
        """Check if given int is odd"""
        return val % 2 != 0

    def __genes_init(self) -> list:
        """Random init Genes

        Generates 5 random genes following these rules:
            - There will be only 5 genes
            - Genes are 2 types: Rotation or CNOT
            - First gene must be a rotation (like last gene)
            - Even indexed genes must be a Rotation
            - Odd indexed genes must be a CNOT

        Returns:
            - the 5 generated genes
        """
        logger.debug("INIT GENES")
        genes = []
        for i in range(Individual.GENE_SIZE):
            actual_gene_type = self.__get_gene_type(i)
            possible_choices = tuple(
                Individual.GENES_MAPPING[actual_gene_type].keys())
            actual_gene = random.choice(possible_choices)

            logger.debug("   GENE %s: type: %s | %s", i,
                         actual_gene_type, actual_gene)
            genes.append(actual_gene)

        return genes

    def __genes_to_qc(self) -> QuantumCircuit:
        """Converts genes to Quantum Circuit

        Each gene is converted to the relative Quantum Circuit (using TwoLocal class), then
        they are merged together into a final Quantum Circuit.

        Returns:
            - Quantum Circuit associated to genes
        """
        logger.debug("GENES TO QC")

        qcs = []
        # common args to pass to TwoLocal class
        common_args = {
            "reps": 1,
            "insert_barriers": True,
            "skip_final_rotation_layer": True,
        }
        for i, gene in enumerate(self.genes):
            actual_gene_type = self.__get_gene_type(i)

            # specific arguments for CNOT or R gene
            _args = {"num_qubits": self.qubits}
            if actual_gene_type == "CNOT":
                _args["entanglement_blocks"] = "cx"
                _args["entanglement"] = Individual.GENES_MAPPING[actual_gene_type][gene]
            else:
                _args["rotation_blocks"] = Individual.GENES_MAPPING[actual_gene_type][
                    gene
                ]

            # merge dicts
            _args = _args | common_args
            logger.debug("   TwoLocal ARGS: %s", _args)

            tmp_qc = TwoLocal(**_args)
            # TODO: print single qc ??

            qcs.append(tmp_qc)

        resulting_qc = QuantumCircuit(self.qubits)
        for i, qc in enumerate(qcs):
            # ENTANGLEMENT BLOCK
            if self.__is_odd(i):
                # add a barrier to CNOT gates. Those are the only
                # ones not ending with a barrier. Its needed to properly
                # separate (visually) circuit's block
                qc.barrier()

            # ROTATION BLOCK
            else:
                if args.qc_params == "all":
                    # replace default param to avoid name conflict while merging
                    new_params = ParameterVector(f"θ{i}", qc.num_parameters)
                    param_map = dict(zip(qc.parameters, new_params))
                    qc = qc.assign_parameters(param_map)

                elif args.qc_params == "one_per_block":
                    # Use a single parameter for each Rotation block
                    rotation_param = Parameter(f"θ{i}")
                    param_dict = {}
                    for param in qc.parameters:
                        param_dict[param] = rotation_param

                    # Replace the original parameters with the shared parameter
                    qc = qc.assign_parameters(param_dict)

            resulting_qc = resulting_qc.compose(qc)

        logger.debug("   \n%s", resulting_qc.decompose().draw())

        return resulting_qc

    def __repr__(self):
        return f"{self.genes}\n{str(self.qc.decompose().draw())}"
