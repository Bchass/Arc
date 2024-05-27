[![check-build](https://github.com/Bchass/Arc/actions/workflows/test-actions.yml/badge.svg)](https://github.com/Bchass/Arc/actions/workflows/test-actions.yml)
[![codecov](https://codecov.io/gh/Bchass/Arc/branch/main/graph/badge.svg?token=gPGfS15fwA)](https://codecov.io/gh/Bchass/Arc)

<h1 align="center">
Arc
</h1>

In the realm of scientific computing, efficiency and simplicity are often paramount. Inspired by the principles of projects like SciPy, Arc emerges as a lightweight alternative, meticulously crafted without the burden of extra dependencies.

Basic usage:

> #### [tinynumpy](https://github.com/Bchass/tinynumpy) is optional
---

```python3
>>> from arc.matrices import csr_matrix
>>> from tinynumpy import tinynumpy as tnp

>>> row1 = tnp.array([0, 0, 1, 1, 2, 2])
>>> col1 = tnp.array([0, 1, 0, 1, 0, 1])
>>> data1 = tnp.array([1, 2, 3, 4, 5, 6])
>>> A = csr_matrix(3, 2, data=data1, row=row1, col=col1)

>>> row2 = tnp.array([0, 0, 1, 1])
>>> col2 = tnp.array([0, 1, 0, 1])
>>> data2 = tnp.array([7, 8, 9, 10])
>>> D = csr_matrix(2, 2, data=data2, row=row2, col=col2)

>>> A.multiply(D).toarray()

array([[25, 28],
       [57, 64],
       [89, 100]], dtype='int64')
```

Example Baseline Benchmark:
---

Arc
```
Profiling Arc implementations:
Arc implementation time for dia_matrix: 0.25066779099870473
         2031 function calls (2030 primitive calls) in 0.251 seconds
```

```python3

from arc.dia import dia_matrix
from arc.data_types import int8
dia_matrix(400, 400, dtype=int8).toarray()
```

SciPy

```
Profiling Scipy implementations:
Scipy implementation time for dia_matrix: 0.1815791659973911
         211031 function calls (211030 primitive calls) in 0.182 seconds
```

```python3
import numpy as np
from scipy.sparse import dia_matrix
dia_matrix((400, 400), dtype=np.int8).toarray()
```
---

[Contributing Guidelines](https://github.com/Bchass/Arc/blob/main/CONTRIBUTING.md)
---

Roadmap:
---
Test cases can be enhanced; additional classes will be offered; push to pypi when there's enough to offer.
