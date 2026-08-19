"""Regression test for tools/build/pack.sh's two lzma-stream levers.

2026-08-14 (eb8897c), pack.sh gained two levers -- `--no-hoist-literals` and
a strip of the payload's leading polyglot shebang -- measured as wins across
every packed family (classic, nnue, the shipped 4k entry, replnet: -22 to
-52 bytes). That commit landed on the nnue-4k branch only and was never
ported back to master, so master's copy of pack.sh silently regressed to
the pre-2026-08-14 spelling: 45 bytes bigger on the shipped 4k entry
(nnue_4k/pst_entry.py at d0a6e60 packed 3455 B / sha ee78e3ca... instead of
3410 B / sha bf30904d...).

Nothing caught it, because check_entry.sh asserts a 4096-byte CEILING, not a
pinned size or sha (see that file's own comment on why: a pin would turn a
routine pyminify/xz version bump into a red CI with nothing wrong). A silent
few dozen bytes is real money in a 4096-byte budget, so this file guards the
WIRING directly -- the two levers must be present in pack.sh's actual
pyminify invocation -- rather than a byte count that would itself rot with
the next toolchain release.
"""

import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACK_SH = ROOT / "tools" / "build" / "pack.sh"
PACK_SH_SRC = PACK_SH.read_text()

# The exact current (fixed) and pre-2026-08-14 (regressed-on-master, 0b41a20)
# pyminify invocations, verbatim from git history. Used both to assert the
# fix is present and, in the end-to-end test below, to reconstruct the
# regressed script for a same-toolchain A/B comparison.
FIXED_PYMINIFY_BLOCK = (
    "pyminify --rename-globals --remove-literal-statements --no-hoist-literals \\\n"
    "   <(sed -e '1{' -e '/^#!/d' -e '}' \\\n"
    "         -e '/# minifier-hide start/,/# minifier-hide end/d' \"$1\") \\\n"
)
REGRESSED_PYMINIFY_BLOCK = (
    "pyminify --rename-globals --remove-literal-statements \\\n"
    "   <(sed '/# minifier-hide start/,/# minifier-hide end/d' \"$1\") \\\n"
)


def test_hoist_literals_lever_present():
    """`--no-hoist-literals` must be on the pyminify invocation.

    Without it, pyminify replaces each repeated string literal with a fresh
    one-character global -- shorter TEXT, but it destroys exactly the
    repetition lzma would otherwise match for free, so the ARTIFACT grows.
    """
    assert "--no-hoist-literals" in PACK_SH_SRC


def test_shebang_strip_lever_present():
    """Line 1 must be dropped when it is a shebang, before minification.

    The polyglot `#!/bin/sh` header is dead inside the artifact -- the
    self-extracting head execs a NAMED interpreter, so nothing ever reads
    the payload's own shebang -- and leaving it in lands the compressed
    stream in a worse lzma neighbourhood.
    """
    assert "1{" in PACK_SH_SRC and "/^#!/d" in PACK_SH_SRC, (
        "expected a sed program that deletes a leading '#!' line (line 1 only)"
    )


def test_fixed_spelling_matches_the_measured_one():
    """The exact, cross-family-measured spelling must be intact.

    Pins the proven wiring (not just the two flags in isolation) so a
    reformat that keeps both levers but changes their interaction --
    e.g. scoping the shebang strip to every line instead of just line 1 --
    fails here instead of shipping unmeasured.
    """
    assert PACK_SH_SRC.count(FIXED_PYMINIFY_BLOCK) == 1
    assert REGRESSED_PYMINIFY_BLOCK not in PACK_SH_SRC


@pytest.mark.skipif(
    not (shutil.which("pyminify") and shutil.which("xz") and shutil.which("bash")),
    reason="pyminify/xz/bash not on PATH",
)
def test_levers_shrink_a_real_pack(tmp_path):
    """End-to-end: today's pack.sh must beat the regressed spelling it replaced.

    Reconstructs the master-regressed invocation by substituting the old
    spelling back into a scratch copy of the real pack.sh (identical
    header logic and everything else -- only the lever line changes),
    packs the same fixture through both, and asserts the real script wins.

    A same-toolchain A/B, not a pinned byte count: it holds regardless of
    the installed pyminify/xz version, which a pinned number would not
    (that is also why check_entry.sh checks a ceiling, not a pin).
    """
    regressed_src = PACK_SH_SRC.replace(FIXED_PYMINIFY_BLOCK, REGRESSED_PYMINIFY_BLOCK)
    regressed_pack_sh = tmp_path / "regressed_pack.sh"
    regressed_pack_sh.write_text(regressed_src)
    regressed_pack_sh.chmod(0o755)

    # A tiny fixture that exercises both levers: a leading polyglot shebang
    # (the same shape sunfish.py itself uses), and a string literal repeated
    # across several functions -- enough for pyminify to hoist it.
    fixture = tmp_path / "fixture.py"
    fixture.write_text(
        "#!/bin/sh\n"
        '""":"\n'
        'exec python3 "$0" "$@"\n'
        '":"""\n'
        "def a():\n"
        '    return "hello world" + "hello world"\n'
        "def b():\n"
        '    return "hello world" + "hello world"\n'
        "def c():\n"
        '    return "hello world" + "hello world"\n'
        "def d():\n"
        '    return "hello world" + "hello world"\n'
        "print(a(), b(), c(), d())\n"
    )

    fixed_out = tmp_path / "fixed.packed"
    regressed_out = tmp_path / "regressed.packed"
    subprocess.run(
        ["bash", str(PACK_SH), str(fixture), str(fixed_out)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["bash", str(regressed_pack_sh), str(fixture), str(regressed_out)],
        check=True, capture_output=True, text=True,
    )

    fixed_size = fixed_out.stat().st_size
    regressed_size = regressed_out.stat().st_size
    assert fixed_size < regressed_size, (
        f"today's pack.sh ({fixed_size} B) should beat the regressed "
        f"spelling ({regressed_size} B) it replaced"
    )

    # The shebang must be gone from the shipped payload, not just shorter:
    # slice out the payload past the self-extracting head (whose own source
    # names the "tail -c+N" offset) and decompress it directly.
    header = fixed_out.read_bytes()[:200].decode("ascii", errors="replace")
    offset = re.search(r"tail -c\+(\d+)", header)
    assert offset, "could not find the self-extracting header's tail offset"
    payload = fixed_out.read_bytes()[int(offset.group(1)) - 1:]  # tail -c +N is 1-indexed
    decompressed = subprocess.run(
        ["xz", "-d"], input=payload, capture_output=True, check=True,
    ).stdout
    assert not decompressed.startswith(b"#!"), "payload still carries a live shebang"
