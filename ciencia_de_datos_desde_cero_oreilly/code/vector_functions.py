from typing import List
from functools import reduce

# Definition of a Vector as a list of numbers
Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:

    # We cannot sum vectors of different lengths
    if len(v) != len(w):
        raise ValueError("Vectors must be the same length")

    # Initialize the output
    output = [0] * len(v)

    # Add Componentwise
    for i in range(len(v)):
        output[i] = v[i] + w[i]

    return output


def subtract(v: Vector, w: Vector) -> Vector:

    # We cannot subtract vectors of different lengths
    if len(v) != len(w):
        raise ValueError("Vectors must be the same length")

    # Initialize the output
    output = [0] * len(v)

    # Subtract Componentwise
    for i in range(len(v)):
        output[i] = v[i] - w[i]

    return output


def sum_of_vectors(vectors: List[Vector]) -> Vector:

    # All vectors must have the same length, we can compare the length of
    # every vector with the first one (for example)
    all_vector_have_the_same_length = all(
        [len(v) == len(vectors[0]) for v in vectors]
    )
    if not all_vector_have_the_same_length:
        raise ValueError(
            "Not all the vectors in the list have the same length"
        )

    # Initialize the output
    output = [0] * len(vectors[0])

    # Add component wise with all the vectors
    for each_vector in vectors:
        output = add(output, each_vector)

    return output


def sum_of_vectors_with_reduce(vectors: List[Vector]) -> Vector:
    output = reduce(add, vectors)
    return output
