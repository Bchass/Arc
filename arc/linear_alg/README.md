- BLAS.py is a wrapper for the Basic Linear Algebra Subprograms (BLAS) library.

- Currently it supports LEVEL 1 functions (not many): https://www.netlib.org/blas/#_history

- Implementation: https://developer.apple.com/documentation/accelerate

Example:

```
>>> from arc.linear_alg.BLAS import arc_BLAS
>>> x = [1.0,2.0,3.0]
>>> y = [4.0,5.0,6.0]
>>> arc_BLAS.sdot(x,y)
32.0
```


NOTE: This is only supported on `macOS` at the moment
