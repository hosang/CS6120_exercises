#!/use/bin/env python3

import argparse
import collections
import dataclasses
import functools
import json
import sys
from typing import Any, Callable, Iterator, Sequence, TypeAlias, TypeVar

Instruction: TypeAlias = dict[str, Any]


@dataclasses.dataclass
class BasicBlock:
  instrs: list[Instruction]
  label: str | None = None
  preds: list[str] = dataclasses.field(default_factory=list)
  succs: list[str] = dataclasses.field(default_factory=list)


def group_instructions(
  instrs: list[Instruction],
) -> Iterator[list[Instruction]]:
  block = []
  for instr in instrs:
    match instr:
      case {'op': 'ret' | 'jmp' | 'br'}:
        block.append(instr)
        if block:
          yield block
        block = []
      case {'label': _}:
        if block:
          yield block
        block = [instr]
      case _:
        block.append(instr)
  if block:
    yield block


def form_blocks(instrs: list[Instruction]) -> list[BasicBlock]:
  blocks = [BasicBlock(instrs=block) for block in group_instructions(instrs)]
  for block_idx, b in enumerate(blocks):
    if 'label' in b.instrs[0]:
      b.label = b.instrs[0]['label']
    else:
      b.label = f'block_{block_idx}'
  return blocks


def update_preds_succs(blocks: list[BasicBlock]) -> list[BasicBlock]:
  for block_idx, b in enumerate(blocks):
    match b.instrs[-1]:
      case {'op': 'jmp', 'labels': [target]}:
        b.succs.append(target)
      case {'op': 'br', 'labels': labels}:
        b.succs.extend(labels)
      case {'op': 'ret'}:
        b.succs = []
      case _:
        if block_idx + 1 < len(blocks):
          # fallthrough
          b.succs.append(blocks[block_idx + 1].label)

  blocks_by_label = {b.label: b for b in blocks if b.label is not None}
  for cur_block in blocks:
    for succ_name in cur_block.succs:
      succ_block = blocks_by_label[succ_name]
      succ_block.preds.append(cur_block.label)
  return blocks


WorklistSet = TypeVar('WorklistSet')


def worklist(
  *,
  blocks: list[BasicBlock],
  init: WorklistSet,
  merge: Callable[[Sequence[WorklistSet]], WorklistSet],
  transfer: Callable[[BasicBlock, WorklistSet], WorklistSet],
  forward: bool = True,
) -> tuple[dict[str, WorklistSet], dict[str, WorklistSet]]:
  if forward:
    in_edges = {b.label: b.preds for b in blocks}
    out_edges = {b.label: b.succs for b in blocks}
  else:
    in_edges = {b.label: b.succs for b in blocks}
    out_edges = {b.label: b.preds for b in blocks}

  outgoing = collections.defaultdict(lambda: init)
  incoming = {}
  block_by_label = {b.label: b for b in blocks if b.label is not None}
  worklist = list(blocks)
  while worklist:
    curr = worklist.pop()
    incoming[curr.label] = merge(outgoing[i] for i in in_edges[curr.label])
    new_outgoing = transfer(curr, incoming[curr.label])
    if new_outgoing != outgoing[curr.label]:
      outgoing[curr.label] = new_outgoing
      worklist.extend(block_by_label[n] for n in out_edges[curr.label])

  if forward:
    return incoming, outgoing
  else:
    return outgoing, incoming


def sort_postorder(blocks: list[BasicBlock]) -> list[BasicBlock]:
  succ_by_label = {b.label: b.succs for b in blocks}
  block_by_label = {b.label: b for b in blocks}
  visited = set()

  def recurse(label: str):
    if label in visited:
      return
    visited.add(label)
    for succ in succ_by_label[label]:
      yield from recurse(succ)
    yield block_by_label[label]

  return list(recurse(blocks[0].label))


def _intersect_pred_doms(sets: Sequence[set]) -> set:
  sets = list(sets)
  if sets:
    return functools.reduce(set.intersection, sets)
  else:
    return set()


def find_dominators(orig_blocks: BasicBlock) -> dict[str, list[str]]:
  blocks = list(reversed(sort_postorder(orig_blocks)))
  all_blocks = {b.label for b in blocks}
  _, dom = worklist(
    blocks=blocks,
    init=all_blocks,
    merge=_intersect_pred_doms,
    transfer=lambda block, inc: inc | {block.label},
  )
  # Put them back into the original order.a
  dominators = {b.label: sorted(dom[b.label]) for b in orig_blocks}
  return dominators


def _analyze(func: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
  blocks = form_blocks(func['instrs'])
  blocks = update_preds_succs(blocks)
  match args.analysis:
    case 'dom':
      return find_dominators(blocks)
    case _:
      raise NotImplementedError(f'Unknown analysis: {args.analysis}')


def main():
  parser = argparse.ArgumentParser(description='Dominance analysis')
  parser.add_argument(
    'analysis',
    choices=['dom', 'tree', 'front'],
    help='The analysis to perform.',
  )
  args = parser.parse_args()

  prog = json.load(sys.stdin)
  result = {}
  for func in prog.get('functions', []):
    result[func['name']] = _analyze(func, args)
  json.dump(result, sys.stdout, indent='  ')


if __name__ == '__main__':
  main()
