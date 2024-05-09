### Arc

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

#### Make Arc faster:

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
>>> from arc.data_types import int8
>>> A = dig_matrix(3,4,dtype=int8).toarray()
>>> print(A)
>>> array ([[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]], dtype=int8)
```

Results:
```
Scipy: 0.03981670900247991
Arc: 0.0021937908604741096
```
