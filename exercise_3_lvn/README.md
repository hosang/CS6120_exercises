# Exercise 1: Local Value Numbering

## Tasks

* Implement trivial dead code elemination (dce).
* Implement the baseic common subexpression elemination (cse) with local value numbering (lvn).
* Optionally implement constant folding, constant propagaion, copy propagaion, and make it invariant to commutative operations.

## Run

Run the tests:

```bash
turnt */*.bril
```

Run benchmarks and print the improvement:

```bash
brench brench.toml -j 8 > results.csv
uv run print_results.py
```
