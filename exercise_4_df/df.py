#!/use/bin/env python3

import argparse
import collections
import dataclasses
import json
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
        b.succs = ['<exit>']
      case _:
        if block_idx + 1 < len(blocks):
          # fallthrough
          b.succs.append(blocks[block_idx + 1].label)
  blocks[0].preds = ['<entry>']
  blocks[-1].succs = ['<exit>']

  blocks_by_label = {b.label: b for b in blocks if b.label is not None}
  for cur_block in blocks:
    for succ_name in set(cur_block.succs) - {'<exit>'}:
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


def get_definitions(block: BasicBlock) -> InstrSet:
  defs = collections.defaultdict(list)
  for instr in block.instrs:
    match instr:
      case {'dest': var_name}:
        defs[var_name].append(instr)
      case _:
        pass
  return defs


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
) -> None:
  def fmt_instrs(
    b: BasicBlock, i_by_block: dict[str, list[Instruction]]
  ) -> str:
    insts = i_by_block[b.label]
    if insts:
      return ', '.join(sorted(str(instr) for instr in insts))
    else:
      return '∅'

  for b in blocks:
    print(f'    .{b.label}:')
    print('        in: ', fmt_instrs(b, inputs))
    print('       out: ', fmt_instrs(b, outputs))


def reaching_definition_analysis(
  args: list[dict[str, str]], blocks: list[BasicBlock]
) -> None:
  block0_inputs = {arg['name']: [arg] for arg in args}
  merge = lambda sets: union(*sets)
  transfer = lambda block, ins: union(
    get_definitions(block), minus(ins, get_kills(block, ins))
  )
  inputs, outputs = forward_worklist(blocks, block0_inputs, merge, transfer)
  print_inputs_outputs(inputs, outputs, blocks)


def live_variables_analysis(
  args: list[dict[str, str]], blocks: list[BasicBlock]
) -> None:
  return_vars = collections.defaultdict(list)
  for b in blocks:
    for instr in b.instrs:
      match instr:
        case {'op': 'ret', 'args': arg_names}:
          for arg_name in arg_names:
            return_vars[arg_name].append(instr)

  merge = lambda sets: union(*sets)

  def transfer(block: BasicBlock, outs: InstrSet) -> InstrSet:
    ins = outs.copy()
    for instr in reversed(block.instrs):
      if 'dest' in instr:
        if instr['dest'] in ins:
          del ins[instr['dest']]
      if 'args' in instr:
        for arg_name in instr['args']:
          ins[arg_name].append(instr)
    return ins

  inputs, outputs = backward_worklist(blocks, {}, merge, transfer, return_vars)
  print_inputs_outputs(inputs, outputs, blocks)


def forward_worklist(
  blocks: list[BasicBlock],
  init: InstrSet,
  merge: Callable[[Sequence[InstrSet]], InstrSet],
  transfer: Callable[[BasicBlock, InstrSet], InstrSet],
  start_init: InstrSet | None = None,
) -> tuple[dict[str, InstrSet], dict[str, InstrSet]]:
  outputs = collections.defaultdict(lambda: init)
  if start_init is not None:
    outputs['<entry>'] = start_init
  inputs = {}

  blocks_by_label = {b.label: b for b in blocks if b.label is not None}
  worklist = list(blocks)
  while worklist:
    cur_block = worklist.pop()
    inputs[cur_block.label] = merge(outputs[pred] for pred in cur_block.preds)
    new_outputs = transfer(cur_block, inputs[cur_block.label])
    if new_outputs != outputs[cur_block.label]:
      outputs[cur_block.label] = new_outputs
      for succ in set(cur_block.succs) - {'<exit>'}:
        worklist.append(blocks_by_label[succ])
  return inputs, outputs


def backward_worklist(
  blocks: list[BasicBlock],
  init: InstrSet,
  merge: Callable[[Sequence[InstrSet]], InstrSet],
  transfer: Callable[[BasicBlock, InstrSet], InstrSet],
  terminal_init: InstrSet | None = None,
) -> tuple[dict[str, InstrSet], dict[str, InstrSet]]:
  inputs = collections.defaultdict(lambda: init)
  if terminal_init is not None:
    inputs['<exit>'] = terminal_init
  outputs = {}

  blocks_by_label = {b.label: b for b in blocks if b.label is not None}
  worklist = list(blocks)
  while worklist:
    cur_block = worklist.pop()
    outputs[cur_block.label] = merge(inputs[succ] for succ in cur_block.succs)
    new_inputs = transfer(cur_block, outputs[cur_block.label])
    if new_inputs != inputs[cur_block.label]:
      inputs[cur_block.label] = new_inputs
      for pred in set(cur_block.preds) - {'<entry>'}:
        worklist.append(blocks_by_label[pred])
  return inputs, outputs


def analyze(func: dict[str, Any], args: argparse.Namespace) -> None:
  blocks = form_blocks(func['instrs'])
  blocks = update_preds_succs(blocks)
  match args.analysis:
    case 'defined':
      reaching_definition_analysis(func.get('args', []), blocks)
    case 'live':
      live_variables_analysis(func.get('args', []), blocks)
    case _:
      raise ValueError(f'Unknown analysis: {args.analysis}')


def main():
  parser = argparse.ArgumentParser(description='Data flow analysis')
  parser.add_argument(
    'analysis', choices=['defined', 'live'], help='The analysis to perform.'
  )
  args = parser.parse_args()

  prog = json.load(sys.stdin)
  for func in prog.get('functions', []):
    print(f'@{func["name"]}')
    analyze(func, args)


if __name__ == '__main__':
  main()
