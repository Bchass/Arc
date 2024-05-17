<h1 align="center">
Arc
</h1>

In the realm of scientific computing, efficiency and simplicity are often paramount. Inspired by the principles of projects like SciPy, Arc emerges as a lightweight alternative, meticulously crafted to without the burden of extra dependencies.

---

#### Basic usage:

```python3
>>> from arc.dia import dia_matrix

>>> A = dia_matrix(2,2)
>>> A.set_element(1, 0, 2)
>>> A.set_element(0, 0, 3)
>>> A.set_element(1, 1, 1)
>>> A.set_element(0, 1, 1)

>>> D = dia_matrix(2, 2)
>>> D.set_element(0, 0, 2)
>>> D.set_element(1, 1, 4)

>>> A.multiply(D)

[6, 4]
[4, 4]
```

---

#### Baseline Benchmark:

Arc
```
Profiling Arc implementation:
Arc implementation time: 0.25261666695587337
         2027 function calls (2026 primitive calls) in 0.253 seconds
```

```python3

from arc.dia import dia_matrix
from arc.data_types import int8
dia_matrix(400, 400, dtype=int8).toarray()
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
Test cases can be enhanced; additional classes will be offered; `dia_matrix` needs a little more work; and `csr_matrix` is a work in progress that will eventually require a refactor. `arc_BLAS` is an introduction and will take some time to mature.
