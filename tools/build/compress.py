#!/usr/bin/env python3
# compressed.py generator: the "Chess logic" + "Search logic" sections of
# the engine (Move/Position/.../Searcher -- the numbfish compressed.py
# style), as readable Python: original names and indentation, with
# comments, docstrings and blank lines removed, and every if/elif/else/
# for/while whose body is a single simple statement folded onto one
# line (`if x: y`) when that fits in WIDTH columns. Everything before
# the "# Chess logic" section header and from the "# UCI User interface"
# header on is hidden (tables, constants, UCI loop). A reading artifact,
# not a runnable script. Stdlib only, deterministic.
import ast, io, sys, tokenize

WIDTH = 120
SIMPLE = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)  # not foldable bodies

raw = open(sys.argv[1]).read().splitlines(keepends=True)
start = next(i for i, ln in enumerate(raw) if ln.strip() == "# Chess logic") - 1
stop  = next(i for i, ln in enumerate(raw) if ln.strip() == "# UCI User interface") - 1
src = ''.join(raw[start:stop])
tree = ast.parse(src)

drop = set()  # docstrings / bare string statements
for node in ast.walk(tree):
    body = getattr(node, 'body', None)
    if isinstance(body, list):
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
               and isinstance(stmt.value.value, str):
                drop.update(range(stmt.lineno, stmt.end_lineno + 1))

lines = src.splitlines()
for tok in tokenize.generate_tokens(io.StringIO(src).readline):
    if tok.type == tokenize.COMMENT:            # '#' inside strings is not a comment
        row, col = tok.start
        lines[row - 1] = lines[row - 1][:col]
lines = [ln.rstrip() for ln in lines]


def fold(header_row, stmt):
    """Put a one-line simple statement on its header's line if the result fits."""
    if isinstance(stmt, SIMPLE) or stmt.lineno != stmt.end_lineno: return
    if lines[stmt.lineno - 1][:stmt.col_offset].strip(): return   # already on its header's line
    head, body = lines[header_row - 1], lines[stmt.lineno - 1].strip()
    if not head.endswith(':'): return
    if len(head) + 1 + len(body) <= WIDTH:
        lines[header_row - 1] = head + ' ' + body
        drop.add(stmt.lineno)


def header_before(row):
    """The nearest kept non-blank line above `row` (the block header's last line)."""
    while row > 1:
        row -= 1
        if row not in drop and lines[row - 1].strip(): return row


for node in ast.walk(tree):
    if isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
        if len(node.body) == 1: fold(header_before(node.body[0].lineno), node.body[0])
        orelse = getattr(node, 'orelse', [])
        if len(orelse) == 1 and not (isinstance(node, ast.If) and isinstance(orelse[0], ast.If)
                                     and lines[orelse[0].lineno - 1].lstrip().startswith('elif')):
            fold(header_before(orelse[0].lineno), orelse[0])

out = [ln for i, ln in enumerate(lines, 1) if i not in drop and ln]
sys.stdout.write('\n'.join(out) + '\n')
