"""Sunfish modules that ship with the wheel.

`sunfish.py` is the engine; this package is everything an installed
sunfish needs around it. It lives outside `tools/` because `tools/` is a
directory of development scripts that is not installed, and because a
distribution may not claim a top-level import name as generic as
`tools`: the name is taken on PyPI, and a namespace package by that name
merges with every other `tools` on `sys.path`.
"""
