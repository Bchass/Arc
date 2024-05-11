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

#### Small victory of beating Scipy at 100x100. Still more optimization to do for something like 1000x1000.

Arc
```
Profiling Arc implementation:
Arc implementation time: 0.02692158310674131
         2027 function calls (2026 primitive calls) in 0.027 seconds
```

SciPy

```
Profiling Scipy implementation:
Scipy implementation time: 0.17151737492531538
         267947 function calls (265630 primitive calls) in 0.259 seconds
```

---

###  Roadmap:
- ✨optimize further ✨
- Implement csr_matrix
- Refactor test cases