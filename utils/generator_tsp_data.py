import random
import json


def _generate_symmetric_tsp_instance(
    num_nodes: int, min_distance: int = 1, max_distance: int = 100
) -> list[list[int]]:
    """
    Generates a symmetric TSP instance with the given number of nodes
    and distance range. The distance between nodes is symmetric and randomly
    generated within the specified range. The distance from node i to node j
    is the same as from node j to node i.
    The diagonal (distance from a node to itself) is set to 0.

    Args:
        num_nodes (int): Number of nodes in the TSP instance.
        min_distance (int): Minimum distance between nodes.
        max_distance (int): Maximum distance between nodes.

    Returns:
        list[list[int]]: A TSP distance matrix.
    """
    distance_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = random.randint(min_distance, max_distance)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix


def _save_instance_to_file(instance: list[list[int]], filename: str) -> None:
    """
    Saves the TSP instance to a JSON file.

    Args:
        instance (list[list[int]]): The TSP instance to save.
        filename (str): The name of the file to save the instance to.
    """
    with open(filename, "w") as f:
        json.dump(instance, f)


if __name__ == "__main__":
    num_nodes = 11

    instance = _generate_symmetric_tsp_instance(num_nodes=num_nodes)
    _save_instance_to_file(instance, "tsp_instance.json")
