Linear algebra is the branch of mathematics that deals with vector spaces.

## Vectors

Vectors are objects that can be added together, that can be multiplied by an 
scalar.

For us, vectors are points in a finite dimensional space.

Although you might not think of your data as vectors, they are a good way to 
represent numeric data.

For example, if you have heights, weights, and ages of a large number of people,
you can treat your data as three-dimensional vectors (height, weight, age).  

Another example, if you are teaching a class with four exams, you can treat 
student grades as four dimensional vectors (exam1, exam2, exam3, exam4).


The simplest approach is to represent vectors as list of numbers. A list of 
three numbers corresponds to a vector in three-dimensional space.


```python
from typing import List

Vector = List[float]

# Vector of height in cm, weight in kg, age
heigth_weigth_age = [70, 170, 40]

# The grades for each exam, considering that we have 4 exam per year and student
grades = [95, 80, 75, 62]
```