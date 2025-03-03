import funciones_generales


def test_add():
    assert funciones_generales.add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

    try:
        funciones_generales.add([1, 2, 3], [4, 5])
    except Exception as e:
        assert str(e) == "vectors must be the same length"


def test_subtract():
    assert funciones_generales.subtract([1, 2, 3], [4, 5, 6]) == [-3, -3, -3]

    try:
        funciones_generales.subtract([1, 2, 3], [4, 5])
    except Exception as e:
        assert str(e) == "vectors must be the same length"


def test_vector_sum():
    assert funciones_generales.vector_sum(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ) == [12, 15, 18]

    try:
        funciones_generales.vector_sum([[1, 2, 3], [1, 2, 3], [4, 5]])
    except Exception as e:
        assert str(e) == "vectors must be the same length"


def test_scalar_multiply():
    assert funciones_generales.scalar_multiply(c=2, v=[1, 2, 3]) == [2, 4, 6]


def test_vector_mean():
    assert funciones_generales.vector_mean(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ) == [4, 5, 6]
