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

>>> A.multiply(D)

[6, 4]
[4, 4]
```

---

#### Arc beating Scipy at 400x400 diagonal_matrix as the benchmark. Arc starts to fall right behind at 500x500.

Arc
```
Profiling Arc implementation:
Arc implementation time: 0.25261666695587337
         2027 function calls (2026 primitive calls) in 0.253 seconds
```

```python3

from arc.dia import dig_matrix
from arc.data_types import int8
dig_matrix(400, 400, dtype=int8).toarray()
```

SciPy

```
Profiling Scipy implementation:
Scipy implementation time: 0.08977137506008148
         267947 function calls (265630 primitive calls) in 0.328 seconds
```

```python3
import numpy as np
from scipy.sparse import dia_matrix
dia_matrix((400, 400), dtype=np.int8).toarray()
```

---

###  Roadmap:
- ✨optimize further ✨
- Implement csr_matrix
- Refactor test cases