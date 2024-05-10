<h1 align="center">
Arc
</h1>

#### Example:

```python3
>>> from arc.dia import dig_matrix

>>> A = dig_matrix(2,2)
>>> A.set_element(1, 0, 2)
>>> A.set_element(0, 0, 3)
>>> A.set_element(1, 1, 1)
>>> A.set_element(0, 1, 1)

>>> D = dig_matrix(2, 2)
>>> D.set_element(0, 0, 2)
>>> D.set_element(1, 1, 4)

>>> result = A.multiply(D)
>>> print(result)

[6, 4]
[4, 4]
```

#### Arc is faster with a basic example. The goal is to beat SciPy at a larger scale like 100x100

SciPy
```python3
>>> import numpy as np
>>> from scipy.sparse import dia_matrix
>>> dia_matrix((3, 4), dtype=np.int8).toarray()
>>> array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)
```

Arc
```python3
>>> from arc.dia import dig_matrix
>>> from arc.data_types import int8
>>> A = dig_matrix(3,4,dtype=int8).toarray()
>>> print(A)
>>> array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)
```

Results:
```
Scipy: 0.0332047080155462
Arc: 0.0021055841352790594
```
