Sunfish
=======
Sunfish is a simple, but strong chess engine, written in Python, mostly for teaching purposes. Without tables and its simple interface, it takes up just 111 lines of code!

The great clarity of the Sunfish code provides a great platform for experimenting, be it with evaluation functions, chess heuristics or search. Fork it today and see what you can do!

Screenshot
==========

    My move: b8c6
    Visited 68997 nodes.
     r . b q k b n r 
     p p p p p p p p 
     . . n . . . . . 
     . . . . . . . . 
     . . . . P . . . 
     . . . . . . . . 
     P P P P . P P P 
     R N B Q K B N R 
              
    Your move: 

Run it!
=======
Sunfish is self contained in the `sunfish.py` file from the repository. I recommend running it with `pypy` for optimal performance.

It is also possible to run Sunfish as an [XBoard](http://www.gnu.org/software/xboard/)/CECP engine in [PyChess](http://pychess.org), [Arena](http://www.playwitharena.com) or your chess interface of choice. Just add the command `pypy -u xboard.py`, replacing `pypy` with your favourite Python interpreter.

Features
===========
1. Build around the simple, but deadly efficient MTD-bi search algorithm.
2. Filled with classic as well as modern 'chess engine tricks' for simpler and faster code.
3. Easily adaptive evalutation function through Piece Square Tables.
4. Uses standard Python collections and data structures for clarity and efficiency.

Limitations
===========
Sunfish supports castling, en passant, and promotion. It doesn't however do minor promotion or draws of any kind. All input must be done in simple 'two coordinate' notation, as shown in the screenshot.

On the technical side there are a lot of features that could be interesting to add to Sunfish. For performance, the most important might be a piecelist to save the enumeration of all board squares at every move generation. Other performance optimizations could include dedicated check detection, zobrist hashing and a mutable board representation - perhaps based on bitboards.

The evaluation in Sunfish is not very sophisticated. E.g. we don't distinguish between midgame and endgame. Not much selective deepening is done, no threat detection and the like. Finally Sunfish might benefit from a more advanced move ordering, including such things as killer move and SEE.

Why Sunfish?
============
The name Sunfish actually refers to the [Pygmy Sunfish](http://en.wikipedia.org/wiki/Pygmy_sunfish), which is amoung the very few fish to start with the letters 'Py'. Using the name of a fish is a reference to the great giants Stockfish and Rybka.

![alt tag](http://upload.wikimedia.org/wikipedia/commons/6/67/Elassoma_sp.jpg)
