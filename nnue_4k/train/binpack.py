"""Stockfish .binpack reader ADAPTER -- flag-gated scaffold, deliberately loud.

Interface: parse(path, limit, cpmax, split_seed) -> the data.py array tuple,
reachable only via DataCfg(kind="binpack").  Design after nnue-pytorch's
training-data reader (https://github.com/official-stockfish/nnue-pytorch,
lib/nnue_training_data_formats.h): READER ONLY.  Their trainer and quantizer
stay inapplicable here -- our packed constraints (16-bit lanes, folded gains,
sum(G) <= 65534, base-90 ternary payloads) have no counterpart in their
pipeline; the only precedents this repo has ever borrowed from nnue-pytorch
are the loss exponent (~2.6) and the feature factorizer (MEASUREMENTS.md).

NOT IMPLEMENTED, on purpose: binpack is a chained delta format (entry =
packed sfen + score + move + ply + result + rule50, followers reconstructed
by applying the recorded move to the previous position), and no .binpack
corpus exists in this workstream to validate a decoder against.  Shipping an
unvalidated decoder would violate the never-hide-errors rule: a mis-parse
here is silent label corruption, the worst failure class a trainer has.
When a binpack corpus actually enters the lane, implement against
nnue-pytorch's reference and add a cross-check on their own tooling's
decode of the same shard before the first training run reads it.
"""


def parse(path, limit, cpmax, split_seed):
    raise NotImplementedError(
        "binpack reading is scaffolded but not implemented: no .binpack corpus "
        "exists in this lane to validate a decoder against, and an unvalidated "
        "decoder is silent label corruption.  See this module's docstring for "
        "the implementation contract (reader only, after nnue-pytorch)."
    )
