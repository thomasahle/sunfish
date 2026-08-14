"""Encoder zoo + bake-off harness for the replnet payload.

The compression problem is unusual enough to earn its own lane (Thomas:
"this is genuinely a unique compression problem, and you should try many
approaches"):

  * the decoder's OWN Python bytes count against the 4096-byte artifact,
    so every arm is (payload + decoder source), never payload alone;
  * lzma wraps the whole entry, so every encoder is really encoder-then-
    lzma and must be measured through the REAL pack path (pack.sh /
    pack_entry.sh) -- composed figures are banned house-wide, and the
    estimate-vs-measured record says the estimates lose;
  * startup has 60 s and numpy is allowed, so decode compute is nearly
    free -- only bytes and correctness are scarce.

Two container layouts per arm (coordinator amendment, Thomas: "I still
think you probably don't want to lzma compress the trained weights. But
you decide what works best" -- resolved by measurement, both ways, per
arm):

  A JOINT     payload digits inside the single lzma stream with the
              engine source (pack.sh, the current shape).
  B SPLIT     engine lzma'd alone, payload bytes appended RAW; the
              engine reads its own artifact via the SF_A/SF_N head
              (pack_entry.sh, the ledger's self-read mechanism from
              4850894/ffead53 -- superseded for base-3 blobs, revived
              here for entropy-coded arms whose output is
              incompressible by construction).

Every (arm, layout) must round-trip BIT-EXACTLY to the trained
quantization -- qnet.expected_module_data() is the independent mirror the
gate compares against -- and the ranked table reports pack.sh's own byte
counts, decoder-source cost as a delta of two measured artifacts, and
decode wall time.

Usage (one command, all encoders, one net):

    python3 nnue_4k/train/compress/bakeoff.py RUN_OR_PICKLE [...]

export.py --bakeoff runs the same zoo and reports the measured winner.
"""
