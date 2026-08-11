"""Embed the base evaluation tables into a net file.

The engine reads pst (classic piece-square tables incl. piece values,
padded 120-wide) and kend (the bare-king mop-up table) from the net
header: they are eval data exactly like the packed rows, and shipping
them with the net removed ~600 bytes of table literals from the 4k
artifact.  Material-base nets carry flat tables -- the engine needs no
branch.

The canonical source is classic sunfish.py at the repo root, whose
module-level pad/join loop produces exactly the tuples the engine used
to build inline (equality is proven downstream by exact bench-node
match: node counts depend on every table value).

usage: embed_tables.py NET_IN NET_OUT     (either .sfnn or .pickle in,
                                           extension picks the format out)
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
# classic lives at the repo root (two above nnue_4k/packed) or, on the
# bench box's flat training dir, one above -- put both on the path
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
import pnet
import sunfish as classic


def tables_for(base_kind):
    if base_kind == "mat":
        flat = {p: [classic.piece[p]] * 120 for p in classic.piece}
        return flat, flat["K"]
    pst = {p: list(classic.pst[p]) for p in classic.piece}
    pst["K"] = list(classic.K_MID)          # never the swapped state
    return pst, list(classic.K_END)


def embed(d):
    pst, kend = tables_for(d.get("base_kind", "pst"))
    d["pst"], d["kend"] = pst, kend
    return d


def main():
    src, dst = sys.argv[1], sys.argv[2]
    if src.endswith(".sfnn"):
        d = pnet.load_sfnn_dict(src)
        d["nts"] = len(d.get("ts", ()))     # save_sfnn re-derives, keep honest
        d.pop("nts")
    else:
        import pickle
        d = pickle.load(open(src, "rb"))
    pnet.save(dst, embed(d))
    print("embedded base tables (%s) -> %s" % (d.get("base_kind", "pst"), dst))


if __name__ == "__main__":
    main()
