![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/License-GPLv3.0-green)

# Ansatz Optimization using Simulated Annealing in Variational Quantum Algorithms for the TSP

This repository explores **Variational Quantum Algorithms (VQAs)** for the **Traveling Salesman Problem (TSP)** using compact **permutation (Lehmer) encoding** and an **adaptive Ansatz** that evolves via **Simulated Annealing (SA)**. It aims to keep qubit counts at **O($nlogn$)** while searching circuit topologies that maximize the **probability of sampling the optimal tour**.

---

## Executive summary

- **What it does.** Searches the space of **Ansatz** circuits via **SA**, evaluates each candidate with a **VQE** over TSP instances, and keeps the best topology.
- **Why it matters.** A compact permutation encoding shrinks the qubit register and **avoids penalty terms** by never casting the objective into a QUBO.
- **How it works.** A 5-gene "genome" specifies alternating rotation and entanglement blocks. Simulated annealing proposes single-gene mutations; a VQE then estimates the expected tour cost and the empirical **$P_{(opt)}$** (probability of sampling the optimal permutation) directly from circuit samples **no QUBO required**. Candidate updates are accepted via the Metropolis criterion.
---

## Project layout

```
Instances/                # Folder of dataset

src/
  args.py                 # CLI flags 
  Individual.py           # Ansatz genome & circuit builder 
  simulated_annealing.py  # SA loop, logging, artifact saving
  myflow.py               # Thin MLflow helpers 
  logger.py               # Logging config with per-run UUID and files
  uuid.py                 # RUN_UUID generator 
  __init__.py             # Convenience exports
  
  VQA/
    __init__.py           # Convenience exports
    tsp_problem.py        # TSP helpers
    vqa_tsp.py            # VQA runner TSP


utils/
    generator_tsp_data.py # File for create instances
```

## Install

> Python 3.9+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quickstart

Create `run_sa.py`:

```python
from src import simulated_annealing, args
from src.logger import setup_logger

if __name__ == "__main__":
    logger = setup_logger()
    best = simulated_annealing(
        eval_file=args.eval_file,           # Instances/5_cities tsp_instance_0.json
        initial_temp=args.initial_temp,     # 1.0
        cooling_rate=args.cooling_rate,     # 0.9
        max_iterations=args.max_iterations  # 100
    )
    logger.info("Done. Best ansatz:\n%s", best)
```

Or `Run with flags` (examples):

---

```bash
python run_sa.py \
  --eval_file Instances/6_cities/tsp_instance_0.json \
  --max_iterations 250 \
  --initial_temp 1.0 \
  --cooling_rate 0.95 \ 
  --optimizers spsa cobyla \
  --vqa_runs_per_instance 3 \ 
  --vqa_num_tries 10 \
```

---

## Theory & background

### VQE / QAOA and objective estimation

Gate-based variational methods (VQA) prepare a parameterized state and optimize classical parameters. 
VQE minimizes the expectation $E(\theta)=\langle \psi(\theta)\mid \hat{H}\mid \psi(\theta)\rangle$; QAOA alternates mixer Hamiltonians and problem Hamiltonians over $p$ levels and adjusts $(\boldsymbol{\gamma},\boldsymbol{\beta})$ on the problem. 
In this project, the goal of VQE is to find the expected cost of the optimal tour based on the output distribution; we therefore monitor $P_{opt}$, the probability that a sampled bit string decodes the optimal tour.

### Compact permutation encoding

Instead of QUBO style encodings that inflate qubits and require penalty terms, compact permutation/Lehmer coding maps tours to integers $[0 \dots n!-1]$.
For this reason uses only $\lceil \log_2(n!)\rceil = O(n\log n)$ qubits.

### Adaptive ansatz via Simulated Annealing

The 5-gene vector $S=\langle x_0,\dots,x_4\rangle$ encodes an approach that alternates between rotation and entanglement blocks. Simulated annealing (SA) proposes a local mutation to a gene, reconstructs the circuit, evaluates it using the VQA, and accepts it with Metropolis probability $p=\min!\big(1, ... ,e^{-\Delta E/T}\big)$, where $\Delta E = E_{\text{new}}-E_{\text{old}}$ and $T$ follows a geometric cooling schedule.
This optimizes the topology with shallow circuit budgets.

---


## Cite & references

If you use this repository, please cite the accompanying paper:

**Paper**  
*F. Fagiolo, N. Vescera, Ansatz Optimization using Simulated Annealing in Variational Quantum Algorithms for the TSP, 2025. (QAI2025)*

**BibTeX**

```bibtex
@article{Fagiolo-Vescera2025VQA-TSP-SA,
  title   = {Ansatz Optimization using Simulated Annealing in Variational Quantum Algorithms for the TSP},
  author  = {Fabrizio Fagiolo, Nicolò Vescera},
  year    = {2025},
  note    = {},
  url     = {}
}
```

