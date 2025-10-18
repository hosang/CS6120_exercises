import argparse
import collections
import dataclasses
import itertools
import json
import sys
from typing import Any, TypeAlias

import dce

Instruction: TypeAlias = dict[str, Any]


@dataclasses.dataclass
class BasicBlock:
  instrs: list[dict]
  label: str | None = None
  preds: list[str] = dataclasses.field(default_factory=list)
  succs: list[str] = dataclasses.field(default_factory=list)


def group_instructions(instrs):
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


def form_blocks(instrs):
  blocks = [BasicBlock(instrs=block) for block in group_instructions(instrs)]
  for block_idx, b in enumerate(blocks):
    if 'label' in b.instrs[0]:
      b.label = b.instrs[0]['label']
    else:
      b.label = f'block_{block_idx}'
  return blocks


def update_preds_succs(blocks):
  for block_idx, b in enumerate(blocks):
    match b.instrs[-1]:
      case {'op': 'jmp', 'labels': [target]}:
        b.succs.append(target)
      case {'op': 'br', 'labels': labels}:
        b.succs.extend(labels)
      case {'op': 'ret'}:
        pass
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


def delete_unreachable(blocks):
  update_preds_succs(blocks)
  new_blocks = [b for i, b in enumerate(blocks) if i == 0 or b.preds]
  return new_blocks, len(blocks) != len(new_blocks)


def per_block(func):
  def wrapper(blocks):
    new_blocks = []
    changed = False
    for b in blocks:
      nb, nc = func(b)
      new_blocks.append(nb)
      changed = changed or nc
    return new_blocks, changed

  return wrapper


def converge(blocks, func):
  changed = True
  while changed:
    blocks, changed = func(blocks)
  return blocks


def print_instr(instr: Instruction):
  match instr:
    case {'label': label}:
      print(f'.{label}:')
    case {'dest': dest, 'op': op}:
      args = instr.get('args', [])
      print(f'  {dest} = {op} ' + ', '.join(args))
    case {'op': op}:
      args = instr.get('args', [])
      print(f'  {op} ' + ', '.join(args))

  print(f'  {instr=}')


def print_lvn_state(value_to_vnum, vnum_to_names, var_to_vnum):
  vnum_to_vars = collections.defaultdict(list)
  for var, vnum in var_to_vnum.items():
    vnum_to_vars[vnum].append(var)

  print(
    '{:40s}    {:4s}   {:40s}    {:40s}'.format(
      'variables', 'vnum', 'val', 'names'
    )
  )
  for val, vnum in value_to_vnum.items():
    names = ', '.join(vnum_to_names[vnum])
    var_list = ', '.join(vnum_to_vars[vnum])
    print(f'{var_list:40s} -> {vnum:4d} : {val!r:40s} -> {names:40s}')
  print()


def lvn(block: BasicBlock, args: argparse.Namespace) -> None:
  value_to_vnum = collections.defaultdict(itertools.count().__next__)
  # The names for this value in chronological order. We try to use the oldest one.
  # We need to be careful to remove names that have been clobbered.
  vnum_to_names = collections.defaultdict(list)
  # The current value of each variable.
  var_to_vnum = {}
  const_vnum_to_constval = {}

  # Handle "inputs" to the block: map all variables to themselves.
  for instr in block.instrs:
    for arg in instr.get('args', []):
      if arg in var_to_vnum:
        continue
      vnum = value_to_vnum[arg]
      var_to_vnum[arg] = vnum
      vnum_to_names[vnum].append(arg)

  next_var_index_gen = itertools.count()

  def gen_fresh_var_name(old_name):
    return f'__lvn_{block.label}_{old_name}_{next(next_var_index_gen)}'

  if args.verbose:
    print_lvn_state(value_to_vnum, vnum_to_names, var_to_vnum)
  for i, instr in enumerate(block.instrs):
    if args.verbose:
      print_instr(instr)
    args_varnames = instr.get('args', [])
    args_vnum = [var_to_vnum[arg] for arg in args_varnames]
    instr['args'] = [vnum_to_names[vn][0] for vn in args_vnum]
    if 'dest' not in instr:
      continue

    # Figure out the value number of the result of this operation.
    match instr:
      case {'op': 'id'} if args.copy_propagation or args.constant_propagation:
        # Copy propagation: dest = id src
        # We can use the value number of src directly.
        (vnum,) = args_vnum
      case {'op': 'const', 'type': tp, 'value': v}:
        val = ('const', tp, v)
        vnum = value_to_vnum[val]
        const_vnum_to_constval[vnum] = v
      case {'op': 'call'}:
        # Function calls can have side effects, so each call is a new value.
        val = ('call', f'instr_{i}')
        vnum = value_to_vnum[val]
      case {'op': 'alloc'}:
        # Every call to alloc produces a new value.
        val = ('alloc', f'instr_{i}')
        vnum = value_to_vnum[val]
      case _:
        val = (instr['op'], *args_vnum)
        if args.commutative:
          val = canonicalize_args(val)
        if args.constant_folding:
          val = maybe_fold_constants(
            val, instr, const_vnum_to_constval, value_to_vnum
          )
        vnum = value_to_vnum[val]

    if vnum_to_names[vnum]:
      # The value has been computed before and we still have a variable containing it, so we can reuse it.
      if (
        args.constant_propagation
        and (constval := const_vnum_to_constval.get(vnum)) is not None
      ):
        # If it is a constant though, we want to use the const instead.
        new_instr = {
          'dest': instr['dest'],
          'type': instr['type'],
          'op': 'const',
          'value': constval,
        }
      else:
        # Replace by "dest: type = id canonical_name".
        new_instr = {
          'dest': instr['dest'],
          'type': instr['type'],
          'op': 'id',
          'args': [vnum_to_names[vnum][0]],
        }
      # Need to keep the instruction object.
      instr.clear()
      instr.update(new_instr)
    else:
      # First time we see this value, remember it.
      var_to_vnum[instr['dest']] = vnum

      will_be_overwritten = any(
        later_instr.get('dest') == instr['dest']
        for later_instr in block.instrs[i + 1 :]
      )
      if will_be_overwritten:
        # We need a name that will not be overwritten later.
        instr['dest'] = gen_fresh_var_name(instr['dest'])

    var_to_vnum[instr['dest']] = vnum
    # We are about to clobber dest, so it is no longer usable for previous values.
    for vnum_names in vnum_to_names.values():
      if instr['dest'] in vnum_names:
        vnum_names.remove(instr['dest'])
    vnum_to_names[vnum].append(instr['dest'])

    if args.verbose:
      print_lvn_state(value_to_vnum, vnum_to_names, var_to_vnum)


def maybe_fold_constants(val, instr, const_vnum_to_val, value_to_vnum):
  op, *args = val
  if op == 'const' or not all(arg in const_vnum_to_val for arg in args):
    return val

  ops = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a // b,
    'eq': lambda a, b: a == b,
    'ne': lambda a, b: a != b,
    'lt': lambda a, b: a < b,
    'gt': lambda a, b: a > b,
    'le': lambda a, b: a <= b,
    'ge': lambda a, b: a >= b,
    'not': lambda a: not a,
    'and': lambda a, b: a and b,
    'or': lambda a, b: a or b,
  }
  arg_vals = [const_vnum_to_val[vn] for vn in args]
  if op in ops:
    const_result = ops[op](*arg_vals)
    instr['op'] = 'const'
    instr['value'] = const_result
    new_val = ('const', const_result)
    const_vnum_to_val[value_to_vnum[new_val]] = const_result
    return new_val
  else:
    return val


def canonicalize_args(val):
  match val:
    case ('add' | 'mul' | 'eq' | 'and' | 'or' | 'ne', a, b) if a > b:
      return (val[0], b, a)
    case ('lt', a, b) if a > b:
      return ('gt', b, a)
    case ('gt', a, b) if a > b:
      return ('lt', b, a)
    case ('le', a, b) if a > b:
      return ('ge', b, a)
    case ('ge', a, b) if a > b:
      return ('le', b, a)
    case _:
      return val


def optimize(instructions, args):
  blocks = form_blocks(instructions)
  for block in blocks:
    lvn(block, args)
  if args.dead_code_elimination:
    blocks = dce.dce(blocks)
  return list(itertools.chain.from_iterable(b.instrs for b in blocks))


def main():
  parser = argparse.ArgumentParser(description='LVN optimizer')
  parser.add_argument(
    '-v', '--verbose', action='store_true', help='enable debug printing'
  )
  parser.add_argument(
    '-d',
    '--dead_code_elimination',
    action='store_true',
    help='enable dead code elimination',
  )
  parser.add_argument(
    '-f',
    '--constant_folding',
    action='store_true',
    help='enable constant folding',
  )
  parser.add_argument(
    '-P',
    '--constant_propagation',
    action='store_true',
    help='enable constant propagation',
  )
  parser.add_argument(
    '-p',
    '--copy_propagation',
    action='store_true',
    help='enable copy propagation',
  )
  parser.add_argument(
    '-c', '--commutative', action='store_true', help='enable commutative ops'
  )
  args = parser.parse_args()

  prog = json.load(sys.stdin)
  for func in prog.get('functions', []):
    func['instrs'] = optimize(func['instrs'], args)
  json.dump(prog, sys.stdout)


if __name__ == '__main__':
  main()
