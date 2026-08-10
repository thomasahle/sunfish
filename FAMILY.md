The sunfish family keeps growing.
Here is a (very incomplete) list of interesting derivatives:

1. **μSunfish / micropython-uSunfish — fizban99** — It substantially re-engineers Sunfish for MicroPython and ESP32-class hardware: mutable 64-square integer board, preallocated move buffers, bounded TT, compact hashing, three-phase evaluation, mobility, LMR, futility pruning, history/killer heuristics, strength levels, opening book and UCI. It's no longer merely “Sunfish made compatible with MicroPython”; it's an exploration of **how much serious chess-engine machinery fits in a microcontroller**.  Adafruit picked it up in 2026 as well. ([blog.adafruit.com][1])

2. **Numbfish — Dimitrios Danopoulos** — It keeps the recognisable compact Sunfish architecture but replaces much of the positional evaluation with a **NumPy-based incrementally updated NNUE**, including a HalfKP accumulator and TensorFlow Lite inference. Its author reports around 2300 Lichess strength while retaining a ~140-line compressed engine.  Numbfish still gets discussed in computer-chess circles and blogs. ([outskirts.altervista.org][2])

3. **sunfish_rs — Recursing** — A proper **Rust port** rather than a Python translation exercise. Particularly interesting because the port exposes places where Python's representation choices aren't necessarily optimal in Rust, and the README explicitly discusses compactness, iterative search, tapered evaluation and benchmarking. It has also actually run as a Lichess bot.  Your own README calls it roughly 100 Elo stronger than the Python engine. ([awesome.ecosyste.ms][3])

4. **Carnatus — zserge** — A small **Go reimplementation of Sunfish**, accompanied by an unusually good article explaining the process. Rather than mechanically translating Python, the author uses Sunfish as a vehicle for explaining board representation, evaluation and tree search in Go. It is almost exactly the educational downstream use case Sunfish seems uniquely good at producing. ([zserge's blog][4])

5. **Sunfish.js — foo123** — A modern **JavaScript port**. This is interesting less for chess strength and more because it takes Sunfish into browsers/Node-style environments where embedding a tiny engine is substantially easier than bundling Stockfish/WASM. The repository describes itself straightforwardly as a JavaScript port of the Python Sunfish engine. ([awesome.ecosyste.ms][5])

6. **Chess.Mojo — vietanhdev** — A Sunfish-based implementation in **Mojo**, described as the first UCI chess engine in that language. Its roadmap explicitly uses Sunfish as the initial architecture before replacing Python-like data structures, adding NNUE and eventually multithreading. This is a wonderful example of Sunfish being used as a **bootstrap engine for a new programming language**. ([awesome.ecosyste.ms][6])

7. **sunfish.lua — soumith** — A **Lua port**. Soumith Chintala is better known for major deep-learning work, which makes this historically interesting: Sunfish's architecture was sufficiently portable that it made sense as a compact engine in Lua too. I’d investigate this one further if you make a derivatives page—the lineage is old but technically interesting.

8. **Solefish — asandwhich** — A **C++ port of Sunfish**. It is interesting precisely because C++ is the conventional chess-engine language: you can compare how the very Pythonic Sunfish architecture survives when transplanted into a language where mutable representations and lower-level optimization are natural. Contemporary references describe it explicitly as a Python-to-C++ Sunfish port. ([Dragan's Blog][7])

9. **sunfishNNUE — kennyfrc** — Different from Numbfish and from your current tiny NNUE. This version bolts Sunfish onto an **external NNUE probing library**, using Stockfish-style `.nnue` networks. It therefore explores “Sunfish search + industrial NNUE evaluation” rather than trying to make the neural component tiny. ([GitHub][8])

10. **Deepfish — dyth** — One of the earlier attempts at **Sunfish + deep learning**. It is old, but historically important because people were already pointing to “Deepfish (Sunfish + Deep Learning)” when discussing open Python engines. ([Reddit][9]) This is probably one of the projects behind the “developing deep learning chess programs” line in the Sunfish README. ([GitHub][10])

11. **ParallelChessAI** — An experiment using the Sunfish codebase to investigate **parallel search**. This is another direction that your README explicitly mentions as an actual Sunfish use case. The surviving repositories are not especially polished products, but conceptually it's significant: minimal code makes search experiments much easier to isolate. ([GitHub][11])

12. **numworks_usunfish — fizban99** — A calculator-specific branch of the μSunfish lineage. Sunfish running on a **NumWorks calculator** is exactly the sort of strange downstream application worth showcasing. A NumWorks-hosted version explicitly describes itself as a Sunfish-based engine ported to MicroPython. ([my.numworks.com][12]) Even better, users subsequently modified it for **Casio calculators**, showing that this derivative itself has started generating derivatives. ([planet-casio.com][13])

13. **chess-badger2040 — niutech** — Sunfish on a **Badger 2040 e-ink RP2040 device**. This isn't just a language port; it's Sunfish embedded into a physical user interface with severe display, CPU and memory constraints. The project is explicitly described as a chess game based on the MicroPython Sunfish engine. ([awesome.ecosyste.ms][14])

14. **micropython-sunfish — jacklinquan** — An earlier/simpler **MicroPython conversion**. This one is valuable because it shows that the desire to get Sunfish onto microcontrollers predates the much more ambitious μSunfish work. It is specifically listed by MicroPython resource collections as a chess engine for MicroPython. ([awesome.ecosyste.ms][15])

15. **micropython-sunfish — peterhinch** — Another separate MicroPython line, interesting because Peter Hinch is a well-known MicroPython developer. It exposes a Sunfish engine as a reusable MicroPython component/API rather than merely as a standalone chess program. ([git.hubp.de][16])

16. **Vector-Anki-Sunfish — mth75** — Sunfish combined with an **Anki Vector robot**. I rank this relatively highly despite its small size because it illustrates something important about Sunfish: it can act as the decision-making core inside a completely different embodied application rather than being “the application” itself.

17. **chess-genetic-algorithm-sunfish — Daspy11** — Uses Sunfish as an environment for **genetic/evolutionary tuning**. That is one of the more natural research directions for your engine because the PST evaluation has relatively few parameters and the entire engine is easy to clone and mutate.

18. **sunfishDDA — zqigolden** — A Sunfish-derived experiment around **dynamic difficulty adjustment**. I find the problem more interesting than the implementation maturity: instead of asking “how strong can Sunfish become?”, it asks how an engine can adapt itself to the human opponent. That fits surprisingly well with the recent μSunfish work on explicit skill levels.

19. **Blindfold-Sunfish — sjqtentacles** — A modification aimed at **blindfold chess / alternative interaction**. It is a smaller derivative, but interesting because it alters what Sunfish is *for*, rather than merely making the same engine faster.

20. **Moonfish — C lineage** — A small chess engine **inspired by Sunfish but written in C**. The project description explicitly says it is inspired by Sunfish. ([shithub.us][17]) I put it below Solefish because this is more “Sunfish-inspired new engine” than direct port, but architecturally that may actually make it more interesting.

[1]: https://blog.adafruit.com/2026/03/24/an-unofficial-micropython-port-of-the-sunfish-chess-engine/?utm_source=chatgpt.com "An unofficial MicroPython port of the Sunfish Chess Engine « Adafruit Industries – Makers, hackers, artists, designers and engineers!"
[2]: https://outskirts.altervista.org/forum/viewtopic.php?p=64597&utm_source=chatgpt.com "Numbfish 1.0 64 ja - Outskirts CheSS ForuM"
[3]: https://awesome.ecosyste.ms/projects/github.com%2Fthomasahle%2Fsunfish?utm_source=chatgpt.com "https://github.com/thomasahle/sunfish | Ecosyste.ms: Awesome"
[4]: https://zserge.com/posts/carnatus/?utm_source=chatgpt.com "Let's write a tiny chess engine in Go"
[5]: https://awesome.ecosyste.ms/projects?owner=foo123&utm_source=chatgpt.com "Projects in Awesome Lists by foo123 | Ecosyste.ms: Awesome"
[6]: https://awesome.ecosyste.ms/projects/github.com%2Fvietanhdev%2Fchess.mojo?utm_source=chatgpt.com "https://github.com/vietanhdev/chess.mojo | Ecosyste.ms: Awesome"
[7]: https://blog.dragansr.com/2021_06_07_archive.html?utm_source=chatgpt.com "DraganSr: 2021-06-07"
[8]: https://github.com/kennyfrc/sunfishNNUE?utm_source=chatgpt.com "GitHub - kennyfrc/sunfishNNUE: Sunfish, a minimalist python chess engine, now with NNUE · GitHub"
[9]: https://www.reddit.com/r/chessprogramming/comments/gm9o9j?utm_source=chatgpt.com "Any open-source engines written in python-chess?"
[10]: https://github.com/thomasahle/sunfish?utm_source=chatgpt.com "GitHub - thomasahle/sunfish: Sunfish: a Python Chess Engine in 111 lines of code · GitHub"
[11]: https://github.com/HarshithBolar/ParallelChessAI?utm_source=chatgpt.com "GitHub - HarshithBolar/ParallelChessAI: Sunfish: a Python Chess Engine in 111 lines of code · GitHub"
[12]: https://my.numworks.com/python/fizban/usunfish_engine?utm_source=chatgpt.com "fizban/usunfish_engine.py - Python — NumWorks"
[13]: https://www.planet-casio.com/Fr/forums/topic18534-16-mpm-mod-add-ins-math.html?utm_source=chatgpt.com "Forum Casio - MPM : Mod add-ins Math+ par Lephenixnoir · Planète Casio"
[14]: https://awesome.ecosyste.ms/projects?owner=niutech&utm_source=chatgpt.com "Projects in Awesome Lists by niutech | Ecosyste.ms: Awesome"
[15]: https://awesome.ecosyste.ms/projects/github.com%2Fpgnethun%2Fawesome-urls?utm_source=chatgpt.com "https://github.com/pgnethun/awesome-urls | Ecosyste.ms: Awesome"
[16]: https://git.hubp.de/PGNetHun/awesome-urls?utm_source=chatgpt.com "GitHub - PGNetHun/awesome-urls: A curated list of awesome URLs · GitHub"
[17]: https://shithub.us/zamfofex/moonfish/ba40d957a856ab2c1ebdde90492f6e9e0485c2a4/README.md/f.html?utm_source=chatgpt.com "shithub: moonfish"
