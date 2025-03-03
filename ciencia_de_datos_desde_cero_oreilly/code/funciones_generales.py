from typing import List

Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    """Sum of 2 vectors

    Args:
        v (Vector): first vector
        w (Vector): second vector

    Returns:
        Vector: sum of the 2 vectors
    """
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    """Substraction of 2 vectors

    Args:
        v (Vector): first vector
        w (Vector): second vector

    Returns:
        Vector: substraction of the 2 vectors
    """
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors: List[Vector]) -> Vector:
    """Sum of a list of vectors

    Args:
        vectors (List[Vector]): list of vectors

    Returns:
        Vector: sum of the list of vectors
    """
    # Check that all vectors have the same length
    # We take the first element of the list to restrict the length of
    # the vectors
    vector_length = len(vectors[0])
    assert all(
        len(v) == vector_length for v in vectors
    ), "vectors must be the same length"

    return [
        sum([vector[i] for vector in vectors]) for i in range(vector_length)
    ]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiply a vector by a scalar

    Args:
        c (float): scalar
        v (Vector): vector

    Returns:
        Vector: vector multiplied by the scalar
    """
    return [c * v_i for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    """
    Calculate the mean of a list of vectors

    Args:
        vectors (List[Vector]): list of vectors

    Returns:
        Vector: mean of the list of vectors
    """
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))
