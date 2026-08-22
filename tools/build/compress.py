#!/usr/bin/env python3
# compressed.py generator: the "Chess logic" + "Search logic" sections of
# the engine (Move/Position/.../Searcher -- the numbfish compressed.py
# style), as readable Python: original names and indentation, with
# comments, docstrings and blank lines removed. Everything before the
# "# Chess logic" section header and from the "# UCI User interface"
# header on is hidden (tables, constants, UCI loop). A reading artifact,
# not a runnable script. Stdlib only, deterministic.
import ast, io, sys, tokenize

raw = open(sys.argv[1]).read().splitlines(keepends=True)
start = next(i for i, ln in enumerate(raw) if ln.strip() == "# Chess logic") - 1
stop  = next(i for i, ln in enumerate(raw) if ln.strip() == "# UCI User interface") - 1
section = raw[start:stop]

src = ''.join(section)
drop = set()  # docstrings / bare string statements
for node in ast.walk(ast.parse(src)):
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

out = [ln.rstrip() for i, ln in enumerate(lines, 1) if i not in drop and ln.rstrip()]
sys.stdout.write('\n'.join(out) + '\n')
