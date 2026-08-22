#!/usr/bin/env python3
# compressed.py generator, matching numbfish's compressed.py exactly:
# ONLY the engine core -- the class definitions and the Entry
# namedtuple -- with comments, docstrings and blank lines removed and
# the original names and indentation kept. The imports, data tables,
# constants, helpers and UCI loop are hidden, so the file is a reading
# artifact, not a runnable script. Stdlib only, deterministic.
import ast, io, sys, tokenize

src_lines = open(sys.argv[1]).read().splitlines(keepends=True)
kept, hide = [], False
for ln in src_lines:
    if '# minifier-hide start' in ln: hide = True; continue
    if '# minifier-hide end' in ln: hide = False; continue
    if not hide: kept.append(ln)
src = ''.join(kept)
tree = ast.parse(src)

show = set()  # top-level sections kept: class defs + namedtuple assigns
for node in tree.body:
    is_ntuple = (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                 and getattr(node.value.func, 'id', '') == 'namedtuple')
    if isinstance(node, ast.ClassDef) or is_ntuple:
        show.update(range(node.lineno, node.end_lineno + 1))

drop = set()  # docstrings / bare string statements inside what we keep
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

out = [ln.rstrip() for i, ln in enumerate(lines, 1)
       if i in show and i not in drop and ln.rstrip()]
sys.stdout.write('\n'.join(out) + '\n')
