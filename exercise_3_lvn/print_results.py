#!/usr/bin/env python3

import collections


def read_results(filename: str) -> tuple[list[str], dict[str, dict[str, int]]]:
  benchmark_results = collections.defaultdict(dict)
  all_methods = set()
  with open(filename) as f:
    column_names = f.readline().strip().split(',')
    assert column_names == ['benchmark', 'run', 'result']
    for line in f:
      benchmark, run, result = line.strip().split(',')
      benchmark_results[benchmark][run] = result
      all_methods.add(run)

  for benchmark in list(benchmark_results.keys()):
    methods = benchmark_results[benchmark]
    missing_methods = all_methods - set(methods.keys())
    if missing_methods:
      raise ValueError(
        f'Benchmark "{benchmark}" is missing results for methods: {list(missing_methods)}'
      )
    if methods['baseline'] in ('missing', 'timeout'):
      print(
        f'Warning: Benchmark "{benchmark}" has baseline result "{methods["baseline"]}", skipping.'
      )
      del benchmark_results[benchmark]
      continue
    for name in methods:
      methods[name] = int(methods[name])
  return list(all_methods), benchmark_results


def main():
  methods, benchmark_results = read_results('results.csv')
  methods.remove('baseline')

  relative_improvements = {m: [] for m in methods}
  absolute_improvements = {m: [] for m in methods}
  baseline_total_ops = 0
  for instructions_per_prog in benchmark_results.values():
    baseline = instructions_per_prog['baseline']
    baseline_total_ops += baseline
    for m in methods:
      absolute_improvements[m].append(baseline - int(instructions_per_prog[m]))
      relative_improvements[m].append(
        (baseline - int(instructions_per_prog[m])) / baseline
      )

  print()
  print(f'Total number of instructions in benchmarks: {baseline_total_ops:,}')
  print()
  print('Total relative improvements over baseline:')
  for m, improvements in sorted(absolute_improvements.items()):
    total_improvement = -sum(improvements) / baseline_total_ops
    print(f'{m:20s} {total_improvement:7.2%}')
  print()
  print('Average relative improvements over baseline:')
  for m, improvements in sorted(relative_improvements.items()):
    mean_improvement = -sum(improvements) / len(improvements)
    print(f'{m:20s} {mean_improvement:7.2%}')


if __name__ == '__main__':
  main()
