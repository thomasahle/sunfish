![Sunfish logo](https://raw.github.com/thomasahle/sunfish/master/docs/logo/sunfish_large.png)

## Introduction
Sunfish is a simple, but strong chess engine, written in Python. With its simple [UCI](http://wbec-ridderkerk.nl/html/UCIProtocol.html) interface, and removing comments and whitespace, it takes up just 138 lines of code! (`tools/build/clean.sh sunfish.py | wc -l`).
Yet [it plays at ratings above 2000 at Lichess](https://lichess.org/@/sunfish-engine).

There is also a (somewhat) stronger NNUE based sunfish, which you can also [play against on Lichess](https://lichess.org/@/sunfish-nnue-engine).
It's only 4096 bytes for the whole engine, so the neuron network is very small.

Because Sunfish is small and strives to be simple, the code provides a great platform for experimenting. People have used it for testing parallel search algorithms, experimenting with evaluation functions, and developing deep learning chess programs. Fork it today and see what you can do!

The name Sunfish refers to the [Pygmy Sunfish](http://en.wikipedia.org/wiki/Pygmy_sunfish), which is among the very few fish to start with the letters 'Py'. The use of a fish is in the spirit of great engines such as Stockfish, Zappa and Rybka. In terms of Heritage, Sunfish borrows much more from [Micro-Max by Geert Muller](http://home.hccnet.nl/h.g.muller/max-src2.html) and [PyChess](http://pychess.org).

# Play against sunfish!

The simplest way to play against sunfish is:
<pre>
$ <b>pip install sunfish && sunfish</b>
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

(From a repo checkout: `sunfish_ui/fancy.py -cmd ./sunfish.py`. GUIs and tournament managers should use the raw UCI engine: `sunfish-uci`, or `./sunfish.py` from a checkout.)
For a true minimalist experience, first we can "pack" sunfish into a compressed executable (less than 3KB!) and run it directly:
<pre>
$ <b>tools/build/pack.sh sunfish.py packed.sh</b>
Total length: 2953
$ <b>./packed.sh</b>
<b>go wtime 1000 btime 1000 winc 1000 binc 1000</b>
info depth 1 score cp 0 pv d2d4
bestmove d2d4
</pre>
(See the [UCI specification](http://wbec-ridderkerk.nl/html/UCIProtocol.html) for the full set of commands.)

### Troubleshooting

`./sunfish.py` automatically runs with `pypy3` if installed (recommended, much stronger), otherwise `python3`.
If the engine fails to start, run `sunfish_ui/fancy.py` with `-debug` to see the underlying error,
and make sure `python3` is on your PATH. On Windows, `.py` engines are launched
through your current Python interpreter automatically.

### Playing with a graphical interface

It is also possible to run Sunfish with a graphical interface, such as [PyChess](http://pychess.org) or [Arena](http://www.playwitharena.de).

Finally you can [play sunfish now on Lichess](https://lichess.org/@/sunfish-engine) or play against [Recursing's Rust port](https://github.com/Recursing/sunfish_rs),
also [on Lichess](https://lichess.org/@/sunfish_rs), which is about 100 ELO stronger.

### NNUE version

There is an experimental version using an [Efficiently updatable neural network](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network).
It is not yet fast enough to be stronger than classic sunfish, so it lives on the
[nnue branch](https://github.com/thomasahle/sunfish/tree/nnue-mutable-board) until it works well.

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
* [sunfish_rs](https://github.com/Recursing/sunfish_rs) - A Rust port of Sunfish that preserves much of its search architecture while using more native Rust representations.
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
