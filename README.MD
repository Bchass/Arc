<h1 align="center">
Arc
</h1>

In the realm of scientific computing, efficiency and simplicity are often paramount. Inspired by the principles of projects like SciPy, Arc emerges as a lightweight alternative, meticulously crafted to without the burden of extra dependencies.

---

#### Basic usage:

```python3
>>> from arc.matrices import csr_matrix

>>> row1 = [0, 0, 1, 1, 2, 2]
>>> col1 = [0, 1, 0, 1, 0, 1]
>>> data1 = [1, 2, 3, 4, 5, 6]
>>> A = csr_matrix(3, 2, data=data1, row=row1, col=col1)

>>> row2 = [0, 0, 1, 1]
>>> col2 = [0, 1, 0, 1]
>>> data2 = [7, 8, 9, 10]
>>> D = csr_matrix(2, 2, data=data2, row=row2, col=col2)

>>> A.multiply(D)

[25, 28]
[57, 64]
[89, 100]
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
