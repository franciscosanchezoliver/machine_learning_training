
Muy a menudo necesitaremos sumar 2 vectores. Los vectores se suman componente a componente, lo que significa que si tenemos 2 vectores: *v* y *w*, su suma es sencillamente:

$v[0]+w[0],v[1]+w[1],...$

Nota: si los vectores no tienen la misma longitud entonces no se pueden sumar.

Ejemplo: sumar los vectores [1, 2] y [2, 1]

$[1,2] + [2, 1] = [1 + 2, 2 + 1] = [3, 3]$


En este ejercicio podemos ver como se suman vectores y como se representa gráficamente la suma de ellos:

[Suma vectores I](./code/draw_sum_of_vectors_0.py)
[Suma vectores II](./code/draw_sum_of_vectors_1.py)
[Suma vectores III](./code/draw_sum_of_vectors_2.py)

Podemos ver como el orden en el que sumamos los vectores no afecta al resultado:

![[Pasted image 20241115080520.png]]

He hecho una función que suma 2 vectores:
[Funcion para sumar 2 vectores](../code/funciones_generales.py)

