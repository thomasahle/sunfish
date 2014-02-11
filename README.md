Sunfish
=======

Sunfish is a near minimal chess engine, written in Python, mostly for teaching purposes. It prioritises clarity over performance, but can look 6 half moves into 'the future' easily. Sunfish provides a great platform for experimenting, be it with evaluation functions, chess heuristics or search. Fork it today and see what you can do!

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

Limitations
===========
Sunfish supports castling, en passant, and promotion. It doesn't however do minor promotion or draws of any kind. All input must be done in simple 'two coordinate' notation.
