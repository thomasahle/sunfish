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
Sunfish is selfcontained in the `sunfish.py` file of the repository. I recommend running it with `pypy` for optimal perforamnce.

It is also possible to run Sunfish as an XBoard/CECP engine in PyChess or a similar interface. It requires a bit of fiddling with the `test.py` script though.

Features
===========
1. Build around the simple, but deadly efficient MTD-bi search algorithm.
2. Filled with classic as well as modern 'chess engine tricks' for simpler and faster code.
3. Easily adaptive evalutation function through Piece Square Tables.
4. Uses standard Python collections and data structures for clarity and efficiency.

Limitations
===========
Sunfish supports castling, en passant, and promotion. It doesn't however do minor promotion or draws of any kind. All input must be done in simple 'two coordinate' notation, as shown in the screenshot.

On the technical side there are a lot of features that could be interesting to add to Sunfish. For performance, the most important might be a piecelist to save the enumeration of all board squares at every move generation. Other performance optimizations include a reduced use of hashtables and a mutable board representation. Perhaps based on bitboards.

The evaluation in Sunfish is not very sophisticated. E.g. we don't distinguish between midgame and endgame. The search is limited in that no quince search is performed and so we can have horizon effects. Null move pruning is not done, and it is debatable wether it would be safe given our choice of MTD search. Finally Sunfish might benefit from a more advanced move ordering, including such things as killer move and SEE.
