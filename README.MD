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
---

#### Arc is going to be faster with a basic example due to less calls and time. 


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

Profiling:

```
Profiling Arc implementation:
Arc implementation time: 0.0036827081348747015
         11027 function calls (11026 primitive calls) in 0.004 seconds
         
```

SciPy
```python3
>>> import numpy as np
>>> from scipy.sparse import dia_matrix
>>> dia_matrix((3, 4), dtype=np.int8).toarray()
>>> array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)
```

Profiling:

```
Profiling Scipy implementation:
Scipy implementation time: 0.15088954102247953
         266981 function calls (264664 primitive calls) in 0.241 seconds
```

---

### The goal is to beat SciPy at a larger scale like 100x100.

Arc
```
Profiling Arc implementation:
Arc implementation time: 0.38364399992860854
         105027 function calls (105026 primitive calls) in 0.384 seconds
```

SciPy

```
Profiling Scipy implementation:
Scipy implementation time: 0.10029954207129776
         266981 function calls (264664 primitive calls) in 0.277 seconds
```