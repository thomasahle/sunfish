# build/ — packing sunfish into a tiny self-extracting executable

Sunfish's party trick is fitting a working chess engine into a few
kilobytes. The scripts here produce that artifact from `sunfish.py`.

## pack.sh — the full packer

    bash build/pack.sh sunfish.py sunfish.packed

Pipeline:

1. **Strip the development-only code.** Everything between
   `# minifier-hide start` and `# minifier-hide end` markers is deleted
   with `sed`. That removes the `import tools.uci` bridge (the full UCI
   implementation used in development and by the lichess bot), leaving
   the self-contained "tiny" UCI loop at the bottom of `sunfish.py` as
   the packed engine's interface.
2. **Minify** with [`pyminify`](https://pypi.org/project/python-minifier/)
   (`--rename-globals --remove-literal-statements`).
3. **Compress** with `xz`.
4. **Prepend a self-extracting header**: a `/bin/sh` stub that copies its
   own tail (`tail -c +N "$0"`) through `xz -d` into a temp file and
   execs it with `pypy3` when available, else `python3` (mirroring the
   polyglot shebang's interpreter preference). The header's byte length appears inside the
   header itself (the `tail -c +N` offset), so it is computed by a small
   fixed-point loop: re-render the header until its length stops
   changing. The `(sleep 9; rm $T)&` arranges for the extracted temp
   file to clean itself up after the engine has started.

The result is a single executable file: `./sunfish.packed` speaks UCI.

Dependencies: `python-minifier`, `xz`, a `python3` on PATH at runtime.

## clean.sh — minify only

Same strip+minify (no rename), written to stdout. Useful to inspect what
the packed build actually contains, or to measure sizes.

## pack_nnue.sh

Lives on the NNUE branch together with the NNUE engine it packs (it
additionally embeds the pickled network weights).

## CI

The workflow runs `pack.sh` on every push and smoke-tests the produced
executable end-to-end (uciok/readyok/bestmove) — the packed artifact is
a release deliverable, and the pipeline has broken silently before (a
`pyminify` update started stripping the shebang, so the header's
`exec $T` fed Python source to /bin/sh; fixed by exec'ing the
interpreter explicitly).
