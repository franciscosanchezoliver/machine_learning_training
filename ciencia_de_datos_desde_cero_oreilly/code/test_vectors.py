import pytest

from ciencia_de_datos_desde_cero_oreilly.code.vector_functions import (
    add,
    subtract,
    sum_of_vectors,
    sum_of_vectors_with_reduce,
)


class TestVectors:

    def test_add_vectors(self):
        result = add([1, 2], [2, 1])
        assert result == [3, 3]

    def test_substract_vectors(self):
        result = subtract([1, 2], [2, 1])
        assert result == [-1, 1]

    def test_sum_list_of_vectors(self):
        v1 = [1, 2, 3]
        v2 = [2, 1, 2]
        v3 = [3, 0, 1]
        list_of_vectors = [v1, v2, v3]

        expected_output = [6, 3, 6]

        result = sum_of_vectors(list_of_vectors)
        assert result == expected_output

    def test_sum_list_of_vectors_with_reduce(self):
        v1 = [1, 2, 3]
        v2 = [2, 1, 2]
        v3 = [3, 0, 1]
        list_of_vectors = [v1, v2, v3]

        expected_output = [6, 3, 6]

        result = sum_of_vectors_with_reduce(list_of_vectors)
        assert result == expected_output
