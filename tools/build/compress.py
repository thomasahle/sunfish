#!/usr/bin/env python3
# compressed.py generator: the engine as readable Python -- original
# names and indentation kept; the dev-only minifier-hide block,
# comments, docstrings/bare-string statements and blank lines removed
# (the numbfish compressed.py style). Stdlib only, deterministic.
import ast, io, sys, tokenize

src_lines = open(sys.argv[1]).read().splitlines(keepends=True)
kept, hide = [], False
for ln in src_lines:
    if '# minifier-hide start' in ln: hide = True; continue
    if '# minifier-hide end' in ln: hide = False; continue
    if not hide: kept.append(ln)
src = ''.join(kept)

drop = set()  # line ranges of docstrings / bare string statements (incl. the sh polyglot)
for node in ast.walk(ast.parse(src)):
    body = getattr(node, 'body', None)
    if isinstance(body, list):
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
               and isinstance(stmt.value.value, str):
                drop.update(range(stmt.lineno, stmt.end_lineno + 1))

lines = src.splitlines()
for tok in tokenize.generate_tokens(io.StringIO(src).readline):
    if tok.type == tokenize.COMMENT:            # tokenize knows '#' inside strings isn't one
        row, col = tok.start
        lines[row - 1] = lines[row - 1][:col]

out = [ln.rstrip() for i, ln in enumerate(lines, 1) if i not in drop and ln.rstrip()]
sys.stdout.write('\n'.join(out) + '\n')
