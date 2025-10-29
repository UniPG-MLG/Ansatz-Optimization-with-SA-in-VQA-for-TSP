import json
import itertools
import math


class TSP_problem:
    def __init__(self, filename: str) -> None:
        self.distance_matrix: list[list[int]] = self._load_instance(filename)
        self.n_cities = len(self.distance_matrix)

    def _load_instance(self, filename: str) -> list[list[int]]:
        """
        Loads a TSP instance from a JSON file.

        Args:
            filename (str): The name of the file to load the instance from.

        Returns:
            list[list[int]]: The loaded TSP instance.
        """
        with open(filename, "r") as f:
            return json.load(f)

    def _evaluate(self, tour: list[int]) -> int:
        """
        Calculates the total cost of a given tour.

        Args:
            tour (list[int]): A list representing the sequence of nodes in the tour.

        Returns:
            int: The total cost associated with the given tour.
        """
        cost = 0
        for i in range(len(tour)):
            cost += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return cost

    def evaluateb(self, x: str) -> int:
        """
        Evaluates a solution represented as a bit-string.

        Args:
            x (str): The bit-string representing the entire solution.

        Returns:
            int: The cost of the tour corresponding to the decoded permutation.
        """
        # Converts the bit-string to an integer, then to a permutation
        perm = self._int2perm(int(x, 2), len(self.distance_matrix))

        return self._evaluate(perm)

    def _find_best_solution(self) -> tuple[int, list[int]]:
        """
        Finds the best solution for the Traveling Salesman Problem (TSP) by evaluating
        all possible permutations of the nodes.

        Returns:
            tuple:
                - best_cost (int): The cost of the optimal tour.
                - best_tour (list[int]): The sequence of nodes representing the optimal tour.
        """
        n = len(self.distance_matrix)
        best_cost = float("inf")
        best_tour = None

        for perm in itertools.permutations(range(n)):
            current_cost = self._evaluate(perm)
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = perm

        return best_cost, list(best_tour)

    @staticmethod
    def _perm2int(perm: list[int]) -> int:
        """
        Converts a permutation into an integer using Lehmer code and factorial base.

        Args:
            perm (list[int]): The permutation to encode.

        Returns:
            int: Integer representation of the permutation.
        """
        n = len(perm)
        lehmer = []
        items = list(range(n))

        for p in perm:
            idx = items.index(p)
            lehmer.append(idx)
            items.pop(idx)

        # Convert Lehmer code to integer
        x = 0
        for i in range(n):
            x += lehmer[i] * math.factorial(n - i - 1)

        return x

    @staticmethod
    def _int2perm(x: int, n: int) -> list[int]:
        """
        Converts an integer back to a permutation using Lehmer code and factorial base.

        Args:
            x (int): The integer to decode.
            n (int): Length of the permutation.

        Returns:
            list[int]: The decoded permutation.
        """
        l = list(range(n))
        p = []
        while n >= 1:
            r = x % n
            x //= n
            p.append(l[r])
            del l[r]
            n -= 1
        return p

    @staticmethod
    def _perm_to_bitestring(perm: list[int]) -> str:
        """
        Converts a permutation into a binary string using factorial encoding.

        Args:
            perm (list[int]): The permutation to convert.

        Returns:
            str: The binary string representation of the permutation.
        """
        x = TSP_problem._perm2int(perm)
        n = len(perm)
        num_bits = math.ceil(math.log2(math.factorial(n)))
        return format(x, f"0{num_bits}b")

    @staticmethod
    def _bitestring_to_perm(bitestring: str, n: int) -> list[int]:
        """
        Converts a binary string back into a permutation.

        Args:
            bitestring (str): The binary string representation.
            n (int): Length of the permutation.

        Returns:
            list[int]: The decoded permutation.
        """
        x = int(bitestring, 2)
        return TSP_problem._int2perm(x, n)
