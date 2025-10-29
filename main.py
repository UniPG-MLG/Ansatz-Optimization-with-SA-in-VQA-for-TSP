from src import args, setup_logger, simulated_annealing


def main():
    logger = setup_logger()
    logger.debug("ARGS: %s", args)

    best = simulated_annealing(
        eval_file=args.eval_file,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        max_iterations=args.max_iterations,
    )

    print("Best ansatz found:", best.genes)
    print("Fitness (prob_opt):", best.fitness)


if __name__ == "__main__":
    main()
