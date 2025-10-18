import dataclasses
import itertools
import json
import sys


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


@per_block
def remove_unused_assignment_local(block):
  skip_instr = [False] * len(block.instrs)
  last_write_by_var = {}
  last_read_by_var = {}
  for idx, instr in enumerate(block.instrs):
    if (dest := instr.get('dest')) is not None:
      if dest in last_write_by_var and dest not in instr.get('args', []):
        if last_write_by_var[dest] > last_read_by_var.get(dest, -1):
          # previous write was never read
          skip_instr[last_write_by_var[dest]] = True
      last_write_by_var[dest] = idx
    if 'args' in instr:
      for arg in instr['args']:
        last_read_by_var[arg] = idx
  new_instrs = [i for i, skip in zip(block.instrs, skip_instr) if not skip]
  return dataclasses.replace(block, instrs=new_instrs), any(skip_instr)


def remove_unused_global(blocks):
  read_vars = set(
    itertools.chain.from_iterable(
      instr.get('args', []) for b in blocks for instr in b.instrs
    )
  )
  changed = False
  for b in blocks:
    new_instrs = [
      instr
      for instr in b.instrs
      if 'dest' not in instr or instr['dest'] in read_vars
    ]
    if len(new_instrs) != len(b.instrs):
      changed = True
    b.instrs = new_instrs
  return blocks, changed


def remove_unused(blocks):
  blocks, changed1 = remove_unused_assignment_local(blocks)
  blocks, changed2 = remove_unused_global(blocks)
  return blocks, changed1 or changed2


def converge(blocks, func):
  changed = True
  while changed:
    blocks, changed = func(blocks)
  return blocks


def dce(blocks):
  blocks = converge(blocks, delete_unreachable)
  blocks = converge(blocks, remove_unused)
  return blocks


def optimize(instructions):
  blocks = form_blocks(instructions)
  blocks = dce(blocks)
  return list(itertools.chain.from_iterable(b.instrs for b in blocks))


def main():
  prog = json.load(sys.stdin)
  for func in prog['functions']:
    func['instrs'] = optimize(func['instrs'])
  json.dump(prog, sys.stdout)


if __name__ == '__main__':
  main()
