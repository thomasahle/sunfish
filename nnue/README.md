# Sunfish NNUE

This is the experimental [NNUE](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)
version of sunfish. In contrast to the large NNUE in, say, Stockfish, these
networks are around 1.2KB, so sunfish NNUE can still be packed into less than 4KB.
It plays better positionally than classic sunfish, but worse tactically, since
the implementation is still not fast enough. Consider it a work in progress.

Run it with:

    tools/fancy.py -cmd "./sunfish_nnue.py nnue/models/tanh.pickle"

## How were the models trained?

The models in `models/` were trained on the
[CCRL dataset](https://lczero.org/blog/2018/09/a-standard-dataset/) published by
the Leela Chess Zero project. That data is not annotated with engine
evaluations, so training used only the win/draw/loss game outcomes as labels,
with an MSE-style loss similar to AlphaZero. Many compact architectures were
tried; none was yet fast enough to make the NNUE version stronger than classic
sunfish. The original training code has unfortunately been lost — see
[issue #119](https://github.com/thomasahle/sunfish/issues/119) for discussion.
If you want to train your own model, the architecture can be reverse-engineered
from the pickle files and the feature/evaluation code in `sunfish_nnue.py`.
