![Sunfish logo](https://raw.github.com/thomasahle/sunfish/master/docs/logo/sunfish_large.png)

Sunfish is a simple, but strong chess engine, written in Python. With its simple [UCI](http://wbec-ridderkerk.nl/html/UCIProtocol.html) interface, and removing comments and whitespace, it takes up just 136 lines of code! (`tools/build/clean.sh sunfish.py | wc -l`).


There is also a (somewhat) stronger NNUE based sunfish, which you can also [play against on Lichess](https://lichess.org/@/sunfish-nnue-engine).
It's only 4096 bytes for the whole engine, so the neuron network is very small.

Because Sunfish is small and strives to be simple, the code provides a great platform for experimenting. People have used it for testing parallel search algorithms, experimenting with evaluation functions, and developing deep learning chess programs. Fork it today and see what you can do!

The name Sunfish refers to the [Pygmy Sunfish](http://en.wikipedia.org/wiki/Pygmy_sunfish), which is among the very few fish to start with the letters 'Py'. The use of a fish is in the spirit of great engines such as Stockfish, Zappa and Rybka. In terms of Heritage, Sunfish borrows much more from [Micro-Max by Geert Muller](http://home.hccnet.nl/h.g.muller/max-src2.html) and [PyChess](http://pychess.org).

## Play against sunfish!

The easiest way to play against sunfish is [@sunfish-engine on Lichess](https://lichess.org/@/sunfish-engine), where you can also play against the stronger [@sunfish-nnue-engine](https://lichess.org/@/sunfish-nnue-engine) (see more below.)

The second easiest way is to play in your terminal:
<pre>
$ <b>pip install sunfish</b>
$ <b>sunfish</b>

Playing against sunfish 2023.
Do you want to be white or black? <b>black</b>
  1 ♖ ♘ ♗ ♔ ♕ ♗ ♘ ♖
  2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
  3
  4
  5
  6
  7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
  8 ♜ ♞ ♝ ♚ ♛ ♝ ♞ ♜
    h g f e d c b a

Score: 23, nodes: 11752, nps: 13812, time: 0.9
 My move: d4
  1 ♖ ♘ ♗ ♔ ♕ ♗ ♘ ♖
  2 ♙ ♙ ♙ ♙   ♙ ♙ ♙
  3
  4         ♙
  5
  6
  7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
  8 ♜ ♞ ♝ ♚ ♛ ♝ ♞ ♜
    h g f e d c b a

Your move (e.g. c6 or g8h6): <b>Nf6</b>
</pre>

Or, from a repo checkout, just run `sunfish_ui/fancy.py -cmd ./sunfish.py`.

### Using sunfish with GUIs and tournament tools

The engine speaks UCI: point any tool at `sunfish-uci` (after
`pip install sunfish`) or `./sunfish.py` (from a checkout).

* **UCI GUIs** ([Arena](http://www.playwitharena.de),
  [Cute Chess](https://cutechess.com), [PyChess](http://pychess.org),
  BanksiaGUI): add an engine, protocol UCI, command `sunfish-uci`.
* **WinBoard / XBoard**: use the
  [PolyGlot](https://github.com/ddugovic/polyglot) adapter with the
  shipped config [`tools/polyglot.ini`](tools/polyglot.ini)
  (tested with PolyGlot 2.0.4; CI drives a real adapter session).
* **Command-line matches** with
  [fastchess](https://github.com/Disservin/fastchess) or
  [cutechess-cli](https://github.com/cutechess/cutechess):

      fastchess -engine cmd=sunfish-uci name=sunfish \
                -engine cmd=<other> name=other \
                -each proto=uci tc=30+1 -rounds 10 -games 2

  (see [`docs/TESTING.md`](docs/TESTING.md) for the full methodology).


### Troubleshooting

`./sunfish.py` automatically runs with `pypy3` if installed (recommended, much stronger), otherwise `python3`.
If the engine fails to start, run `sunfish_ui/fancy.py` with `-debug` to see the underlying error,
and make sure `python3` is on your PATH. On Windows, `.py` engines are launched
through your current Python interpreter automatically.

### Packing

For a true minimalist experience, sunfish can be packed as a single executable of less than 3kb:
<pre>
$ <b>tools/build/pack.sh sunfish.py packed.sh</b>
Total length: 2953
$ <b>./packed.sh</b>
<b>go wtime 1000 btime 1000 winc 1000 binc 1000</b>
info depth 1 score cp 0 pv d2d4
bestmove d2d4
</pre>

This version uses a [simplified UCI protocol by the TCEC 4k rules](https://wiki.chessdom.org/TCEC_4k_Rules#:~:text=A%204K%20UCI%20protocol%20is%20used).

### NNUE version

[nnue_4k/](nnue_4k/) is a sunfish whose evaluation is classic's exact
piece-square score plus a trained neural residual — with the whole
accumulator and evaluation head packed into **one Python integer**, so a
wide net costs a handful of big-int operations per node. It is measured
about +200 Elo over classic at tournament time controls, the engine
still packs to a few kilobytes, and every quantized net is *certified*
(lane-exactness, incremental == from-scratch, exact antisymmetry) before
it plays a game. See [nnue_4k/README.md](nnue_4k/README.md) for the
architecture, the training pipeline, and the measured results.

# Features

1. Built around the simple, but efficient MTD-bi search algorithm, also known as [C*](https://www.chessprogramming.org/NegaC*).
2. Filled with classic "chess engine tricks" for simpler and faster code.
3. Efficiently updatable evaluation function through [Piece Square Tables](https://www.chessprogramming.org/Piece-Square_Tables).
4. Uses standard Python collections and data structures for clarity and efficiency.

# Testing

Sunfish uses pytest for testing. To run the tests:

```bash
python3.12 -m pytest
```

You can also run specific test files:

```bash
python3.12 -m pytest tests/test_mate_puzzles.py
```

Or even specific tests:

```bash
python3.12 -m pytest tests/test_mate_puzzles.py::test_mate_in_one
```

Make sure you have installed the required dependencies (defined in `pyproject.toml`):

```bash
uv sync  # or: pip install chess tqdm pytest pytest-asyncio
```

# Limitations

Sunfish supports all chess rules, except the 50-move draw rule.

There are many ways in which you may try to make Sunfish stronger. First you could change from a board representation to a mutable array and add a fast way to enumerate pieces. Then you could implement dedicated capture generation, check detection and check evasions. You could also move everything to bitboards, implement parts of the code in C or experiment with parallel search!

The other way to make Sunfish stronger is to give it more knowledge of chess. The current evaluation function only uses piece square tables - it doesn't even distinguish between midgame and endgame. You can also experiment with more pruning - currently only null move is done - and extensions - currently none are used. Finally Sunfish might benefit from a more advanced move ordering, MVV/LVA and SEE perhaps?

An easy way to get a strong Sunfish is to run it with the
[PyPy Just-In-Time interpreter](https://pypy.org/) — the launcher at the top
of `sunfish.py` picks `pypy3` automatically when installed. Measured on the
current engine (fixed-depth battery, identical node counts): **PyPy 3.11
searches ~2.7x faster than CPython 3.14** (81 vs 30 knps), worth on the
order of 100 Elo at fast time controls.

(Historical footnote: sunfish once ran fastest under PyPy 2.7, and an old
version of this table said so. Modern sunfish requires Python >= 3.8 —
the code uses the walrus operator — and modern PyPy 3 has long since
closed the gap.)


# Family

The sunfish family keeps growing.
Here is a (very incomplete) list of interesting derivatives:

* [μSunfish](https://github.com/fizban99/micropython-usunfish) - A heavily reworked MicroPython Sunfish derivative for ESP32-class microcontrollers, with bounded memory, stronger search, configurable skill levels, and UCI support.
* [Numbfish](https://github.com/dimdano/numbfish) - A compact Sunfish-based Python engine that adds an incrementally updated NumPy NNUE evaluation.
* [sunfish_rs](https://github.com/Recursing/sunfish_rs) - A Rust port of Sunfish that preserves much of its search architecture while using more native Rust representations. ([Lichess bot](https://github.com/Recursing/sunfish_rs)).
* [Carnatus](https://github.com/zserge/carnatus) - A small Go port of Sunfish intended to keep the engine minimal but readable.
* [Sunfish.js](https://github.com/foo123/sunfish.js) - A JavaScript port of Sunfish designed to run under Node.js or directly in a browser via a web worker.
* [Chess.Mojo](https://github.com/vietanhdev/chess.mojo) - A Sunfish-based UCI chess engine used as a proof of concept for the Mojo programming language.
* [sunfish.lua](https://github.com/soumith/sunfish.lua) - A direct human translation of the classic Sunfish implementation from Python into Lua.
* [Solefish](https://github.com/asandwhich/solefish) - A C++ port that deliberately follows the original Sunfish program structure closely as a learning exercise.
* [sunfishNNUE](https://github.com/kennyfrc/sunfishNNUE) - A Sunfish derivative that replaces its classical evaluation with NNUE evaluation using Stockfish-style neural networks.
* [Moonfish](https://github.com/walker8088/moonfish) - A Sunfish-derived engine adapted from Western chess to Chinese chess/Xiangqi with UCCI support.
* [ParallelChessAI](https://github.com/HarshithBolar/ParallelChessAI) - A Sunfish-derived experiment investigating parallelization of chess-engine search.
* [numworks_usunfish](https://github.com/fizban99/numworks_usunfish) - A μSunfish adaptation designed to run as a playable chess engine on NumWorks calculators.
* [chess-badger2040](https://github.com/niutech/chess-badger2040) - A standalone offline chess game that embeds a MicroPython Sunfish port on the RP2040-based Badger 2040 e-ink device.
* [micropython-sunfish](https://github.com/jacklinquan/micropython-sunfish) - An early port of Sunfish to MicroPython for running the engine on constrained embedded hardware.
* [peterhinch/micropython-sunfish](https://github.com/peterhinch/micropython-sunfish) - Another independent MicroPython adaptation that makes Sunfish usable as an embedded chess-engine component.
* [Vector-Anki-Sunfish](https://github.com/mth75/Vector-Anki-Sunfish) - A project integrating Sunfish with the Anki Vector robot to let the robot participate in chess.
* [chess-genetic-algorithm-sunfish](https://github.com/Daspy11/chess-genetic-algorithm-sunfish) - An experimental Sunfish derivative using genetic algorithms to evolve or tune chess-engine parameters.
* [sunfishDDA](https://github.com/zqigolden/sunfishDDA) - A Sunfish experiment exploring dynamic difficulty adjustment rather than simply maximizing engine strength.
* [Blindfold-Sunfish](https://github.com/sjqtentacles/Blindfold-Sunfish) - A Sunfish modification aimed at blindfold chess and alternative ways of interacting with the engine.
* [Moonfish](https://moonfish.cc/) - A tiny C chess engine originally inspired by Sunfish that subsequently developed its own substantially different implementation and search approach.

# License

[GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html)
