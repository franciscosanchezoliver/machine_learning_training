Definidos de una forma abstracta, los vectores son objetos que se pueden sumar para formar nuevos vectores y se pueden multiplicar por escalares (números) para formar nuevos vectores.

Para nosotros, podemos decir que los vectores son puntos en un espacio de dimensión finita. 

Aunque no solemos pensar en los datos como vectores, a menudo son una forma útil de representar datos numéricos. 

Por ejemplo, si tenemos: la altura, peso y edad de un gran número de personas, podemos tratar los datos como vectores tridimensionales (height, weigth, age).

O por ejemplo, si tuviésemos una clase con cuatro exámenes, podríamos tratar las notas de cada alumno alumnos como un vector de cuatro dimensiones (exam1, exam2, exam3, exam4).

El enfoque más sencillo sería representar los vectores como una lista de números. 

Una lista de 3 números correspondería a un vector en un espacio tridimensional y viceversa.

```python
from typing import List

Vector = List[float]

# Vector of height in cm, weight in kg, age
heigth_weigth_age = [70, 170, 40]

# The grades for each exam, considering that we have 4 exam per year and student
grades = [95, 80, 75, 62]
```

También queremos poder hacer operaciones con vectores. Pero como las List de Python no son vectores, y por tanto no dan facilidades para la aritmética de vectores entonces tendremos que crear nosotros mismos estas utilidades para hacer aritmética de vectores.
