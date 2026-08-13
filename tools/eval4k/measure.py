"""Price an eval-table encoding: real file, real packer, real bytes.

For each scheme we
  1. splice its source into the entry,
  2. pack it with tools/build/pack.sh and read the size off disk,
  3. compare the decoded tables against the entry's own tables,
  4. time the decode (the TCEC startup budget is 60 s, not infinite),
  5. run the packed artifact ALONE in an empty directory with SF_NET unset.

Step 5 is the one that has caught every fake entry so far, so it is not
optional and its failure is fatal, not a warning.

usage: python3 tools/eval4k/measure.py [scheme ...]
"""
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schemes  # noqa: E402
import splice  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pack(src_text, name):
    d = tempfile.mkdtemp(prefix="eval4k-")
    py = os.path.join(d, name + ".py")
    out = os.path.join(d, name + ".packed")
    open(py, "w").write(src_text)
    r = subprocess.run(["bash", os.path.join(ROOT, "tools/build/pack.sh"), py, out],
                       capture_output=True, text=True, cwd=ROOT)
    if not os.path.exists(out):
        raise RuntimeError("pack failed: " + r.stdout + r.stderr)
    return out, os.path.getsize(out)


def standalone(packed):
    """Run the artifact alone in an empty dir, SF_NET unset. Returns bestmove."""
    d = tempfile.mkdtemp(prefix="eval4k-run-")
    tgt = os.path.join(d, "entry")
    open(tgt, "wb").write(open(packed, "rb").read())
    os.chmod(tgt, 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("SF_NET", "SF_A", "SF_N")}
    r = subprocess.run([tgt], input="uci\nisready\nposition startpos\ngo depth 4\nquit\n",
                       capture_output=True, text=True, cwd=d, env=env, timeout=120)
    left = [f for f in os.listdir(d) if f != "entry"]
    if left:
        raise RuntimeError("artifact left files behind: %r" % left)
    for line in r.stdout.splitlines():
        if line.startswith("bestmove"):
            return line.strip()
    raise RuntimeError("no bestmove; stdout=%r stderr=%r" % (r.stdout[-400:], r.stderr[-400:]))


def decode_time(src_text, name):
    """Wall time of the spliced region alone, in the interpreter that runs it."""
    d = tempfile.mkdtemp(prefix="eval4k-t-")
    _, mid, _ = splice.split(src_text)
    prog = os.path.join(d, "t.py")
    open(prog, "w").write("import time\n_t0=time.perf_counter()\n" + mid +
                          "\nprint('%.4f' % ((time.perf_counter()-_t0)*1000))\n")
    best = 1e9
    for _ in range(3):
        r = subprocess.run(["pypy3", prog], capture_output=True, text=True)
        best = min(best, float(r.stdout.strip().splitlines()[-1]))
    return best


def tables_of(src_text):
    """The padded pst / K_MID / K_END the spliced source actually produces."""
    _, mid, _ = splice.split(src_text)
    ns = {}
    exec(mid, ns)
    return ns["pst"], ns["K_MID"], ns["K_END"]


def main():
    base_src = open(os.path.join(ROOT, splice.ENTRY)).read()
    _, base_out = pack(base_src, "baseline")
    base_pst, base_kmid, base_kend = tables_of(base_src)
    print("baseline entry: %d bytes (%d spare)  bestmove=%s" %
          (base_out, 4096 - base_out, standalone(_)))
    print()
    print("%-22s %6s %7s %8s %9s  %s" % ("scheme", "bytes", "spare", "d_eval", "decode", "tables"))
    want = sys.argv[1:] or list(schemes.SCHEMES)
    for name in want:
        try:
            rep = schemes.SCHEMES[name]()
        except Exception as e:
            print("%-22s BUILD FAIL %s" % (name, e))
            continue
        src = splice.splice(base_src, rep)
        try:
            packed, size = pack(src, name)
            pst, kmid, kend = tables_of(src)
            same = (pst == base_pst and kmid == base_kmid and kend == base_kend)
            if not same:
                d = max(abs(a - b) for k in base_pst for a, b in zip(pst[k], base_pst[k]))
                same = "max|d|=%d" % d
            else:
                same = "EXACT"
            ms = decode_time(src, name)
            bm = standalone(packed)
            assert bm.startswith("bestmove") and len(bm.split()) == 2, bm
            print("%-22s %6d %7d %+8d %7.2fms  %s" %
                  (name, size, 4096 - size, size - base_out, ms, same))
        except Exception as e:
            print("%-22s FAIL: %s" % (name, str(e)[:160]))


if __name__ == "__main__":
    main()
