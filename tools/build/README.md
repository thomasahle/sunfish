# build/ — packing sunfish into a tiny self-extracting executable

Sunfish's party trick is fitting a working chess engine into a few
kilobytes. The scripts here produce that artifact from `sunfish.py`.

## pack.sh — the full packer

    bash tools/build/pack.sh sunfish.py sunfish.packed

Pipeline:

1. **Strip the development-only code.** Everything between
   `# minifier-hide start` and `# minifier-hide end` markers is deleted
   with `sed`. That removes the `import sunfish_ui.uci` bridge (the
   full UCI implementation, used everywhere else — development, the
   lichess bot, and `pip install sunfish`), leaving the self-contained
   "tiny" UCI loop at the bottom of `sunfish.py` as the packed engine's
   interface. The bridge is an unconditional import, not a fallback: the
   packed build is the only configuration that runs the tiny loop, and
   it gets there by deleting the import rather than by failing it.
2. **Drop the polyglot `#!/bin/sh` line** (line 1 only, and only if it is a
   shebang). It is dead weight *inside the artifact*: the header in step 5
   execs a named interpreter, so nothing ever reads the payload's shebang.
   The source file keeps its header — only the copy in the payload goes.
3. **Minify** with [`pyminify`](https://pypi.org/project/python-minifier/)
   (`--rename-globals --remove-literal-statements --no-hoist-literals`).
   Hoisting is off *on purpose*: it shrinks the text and grows the artifact,
   because rewriting each repeated literal to a fresh one-character name
   destroys exactly the repetition lzma compresses for free. Steps 2 and 3
   together are −22 to −52 bytes depending on the engine; measured per family
   in the header comment of `pack.sh`, and neither pays without the other.
4. **Compress** with `xz` (`--format=lzma`, `pb=0`).
5. **Prepend a self-extracting header**: a `bash` stub that feeds its own
   tail (`tail -c +N "$0"`) through `xz -d` and hands the result to `pypy3`
   when available, else `python3` (mirroring the polyglot shebang's
   interpreter preference) as a **process substitution** — a `/dev/fd` path,
   so there is no temp file to create, chmod, or clean up. Naming the
   interpreter explicitly is also what makes step 2 safe. The header's byte
   length appears inside the header itself (the `tail -c +N` offset), so it
   is computed by a small fixed-point loop: re-render the header until its
   length stops changing.

The result is a single executable file: `./sunfish.packed` speaks UCI.

Dependencies: `python-minifier`, `xz`, a `python3` on PATH at runtime.

## clean.sh — minify only

Same strip+minify (no rename), written to stdout. Useful to inspect what
the packed build actually contains, or to measure sizes.

## pack_nnue.sh

Lives on the NNUE branch together with the NNUE engine it packs (it
additionally embeds the pickled network weights).

## The committed artifact and the hook

The packed build of `sunfish.py` is committed at the repo root as
[`compressed.py`](../../compressed.py), so the README links to the
artifact itself rather than a command. `tools/hooks/pre-commit`
regenerates it from the *staged* `sunfish.py` whenever that file is part
of a commit (enable once per clone: `git config core.hooksPath
tools/hooks`), and CI fails any push where the committed artifact's
decompressed payload no longer matches a fresh pack of `sunfish.py`.

## CI

The workflow runs `pack.sh` on every push and smoke-tests the produced
executable end-to-end (uciok/readyok/bestmove), separately checks that
the committed `compressed.py` is current and playable — the packed artifact is
a release deliverable, and the pipeline has broken silently before (a
`pyminify` update started stripping the shebang, so the header's
`exec $T` fed Python source to /bin/sh; fixed by exec'ing the
interpreter explicitly). That fix is what turned the payload's shebang
into dead bytes, which is why step 2 can now delete it on purpose — but
the smoke test is the thing that keeps it true, so do not drop it.
