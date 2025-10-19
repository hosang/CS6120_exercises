#!/use/bin/env python3

import argparse
import collections
import dataclasses
import json
import operator
import sys
from typing import Any, Callable, Iterator, Sequence, TypeAlias

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


InstrSet: TypeAlias = dict[str, list[Instruction]]


def get_kills(block: BasicBlock, defs: InstrSet) -> InstrSet:
  kills = {}
  for instr in block.instrs:
    match instr:
      case {'dest': var_name} if var_name in defs:
        kills[var_name] = defs[var_name]
      case _:
        pass
  return kills


def get_gen(block: BasicBlock) -> InstrSet:
  gens = {}
  for instr in block.instrs:
    match instr:
      case {'dest': var_name}:
        gens[var_name] = [instr]
      case _:
        pass
  return gens


def get_use(block: BasicBlock) -> InstrSet:
  uses = collections.defaultdict(list)
  defined = set()
  for instr in block.instrs:
    if args := instr.get('args', []):
      for var in set(args) - defined:
        uses[var].append(instr)
    if 'dest' in instr:
      defined.add(instr['dest'])
  return uses


def union(*sets: Sequence[InstrSet]) -> InstrSet:
  result = collections.defaultdict(list)
  for s in sets:
    for var_name, instrs in s.items():
      result[var_name].extend(instrs)
  return result


def minus(a: InstrSet, b: InstrSet) -> InstrSet:
  result = collections.defaultdict(list)
  for var_name, instrs in a.items():
    if var_name not in b:
      result[var_name].extend(instrs)
  return result


def print_inputs_outputs(
  inputs: dict[str, InstrSet],
  outputs: dict[str, InstrSet],
  blocks: list[BasicBlock],
  *,
  print_vals: bool = False,
) -> None:
  def fmt_instrs(
    b: BasicBlock, i_by_block: dict[str, list[Instruction]]
  ) -> str:
    instrs = i_by_block[b.label]
    if instrs and print_vals:
      return ', '.join(sorted(f'{k}={sorted(v)}' for k, v in instrs.items()))
    elif instrs:
      return ', '.join(sorted(str(instr) for instr in instrs))
    else:
      return '∅'

  for b in blocks:
    print(f'    .{b.label}:')
    print('        in: ', fmt_instrs(b, inputs))
    print('       out: ', fmt_instrs(b, outputs))


def reaching_definition_analysis(
  args: list[dict[str, str]], blocks: list[BasicBlock]
) -> None:
  merge = lambda sets: union(*sets)
  transfer = lambda block, ins: union(
    get_gen(block), minus(ins, get_kills(block, ins))
  )
  inputs, outputs = worklist(
    blocks=blocks, init={}, merge=merge, transfer=transfer
  )
  print_inputs_outputs(inputs, outputs, blocks)


def live_variables_analysis(
  args: list[dict[str, str]], blocks: list[BasicBlock]
) -> None:
  merge = lambda sets: union(*sets)
  transfer = lambda block, outs: union(
    get_use(block), minus(outs, get_gen(block))
  )

  inputs, outputs = worklist(
    blocks=blocks, init={}, merge=merge, transfer=transfer, forward=False
  )
  print_inputs_outputs(inputs, outputs, blocks)


def _const_prop_merge(*sets: Sequence[InstrSet]) -> InstrSet:
  if not sets:
    return {}
  result = {}
  for consts in sets:
    for var_name, values in consts.items():
      if var_name not in result:
        result[var_name] = values
      elif result[var_name] != values:
        result[var_name] = ['?']
  return result


def _const_prop_constants(block: BasicBlock, in_consts: InstrSet) -> InstrSet:
  consts = in_consts.copy()
  constprop_ops = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': operator.floordiv,
    'eq': operator.eq,
    'ne': operator.ne,
    'lt': operator.lt,
    'le': operator.le,
    'gt': operator.gt,
    'ge': operator.ge,
    'and': operator.and_,
    'or': operator.or_,
  }
  for instr in block.instrs:
    match instr:
      case {'op': 'const', 'dest': var_name, 'value': val}:
        consts[var_name] = [val]
      case {'op': op, 'args': [arg1, arg2], 'dest': dest} if (
        op in constprop_ops and arg1 in consts and arg2 in consts
      ):
        (a,) = consts[arg1]
        (b,) = consts[arg2]
        if '?' in (a, b):
          consts[dest] = ['?']
        else:
          consts[dest] = [constprop_ops[op](a, b)]
      case {'dest': var_name}:
        if var_name in consts:
          del consts[var_name]
  return consts


def const_propagation_analysis(
  args: list[dict[str, str]], blocks: list[BasicBlock]
) -> None:
  inputs, outputs = worklist(
    blocks=blocks,
    init={},
    merge=lambda sets: _const_prop_merge(*sets),
    transfer=lambda block, consts: _const_prop_constants(block, consts),
  )
  print_inputs_outputs(inputs, outputs, blocks, print_vals=True)


def worklist(
  *,
  blocks: list[BasicBlock],
  init: InstrSet,
  merge: Callable[[Sequence[InstrSet]], InstrSet],
  transfer: Callable[[BasicBlock, InstrSet], InstrSet],
  forward: bool = True,
) -> tuple[dict[str, InstrSet], dict[str, InstrSet]]:
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


def analyze(func: dict[str, Any], args: argparse.Namespace) -> None:
  blocks = form_blocks(func['instrs'])
  blocks = update_preds_succs(blocks)
  match args.analysis:
    case 'defined':
      reaching_definition_analysis(func.get('args', []), blocks)
    case 'live':
      live_variables_analysis(func.get('args', []), blocks)
    case 'const':
      const_propagation_analysis(func.get('args', []), blocks)
    case _:
      raise ValueError(f'Unknown analysis: {args.analysis}')


def main():
  parser = argparse.ArgumentParser(description='Data flow analysis')
  parser.add_argument(
    'analysis',
    choices=['defined', 'live', 'const'],
    help='The analysis to perform.',
  )
  args = parser.parse_args()

  prog = json.load(sys.stdin)
  for func in prog.get('functions', []):
    print(f'@{func["name"]}')
    analyze(func, args)


if __name__ == '__main__':
  main()
