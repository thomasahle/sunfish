#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal wrapper class for sunfish
"""

import re
import time
import sunfish as sf

Square = int
Color = bool

WHITE = True
BLACK = False

parse_square = sf.parse
square_name = sf.render

# TODO: actual compatibility with python-chess, not just resemble it's behavior

class GameOver(Exception): pass
# TODO: Game over detecting
# I didn't quite understand how sunfish detects gameover, so I left it unimplemented.
# Sunfish.push() and Sunfish.think() should raise GameOver if the game has ended

class Move:
    def __init__(self, from_square: Square, to_square: Square, side: Color = WHITE):
        self.from_square = from_square
        self.to_square = to_square
        self.side = side

    def __iter__(self):
        # to make unpacking(*) operator work inside sunfish
        return iter((self.from_square, self.to_square))

    def __repr__(self) -> str:
        return f"Move.from_uci('{self.uci()}')"

    def __str__(self) -> str:
        return self.uci()

    def __uci(self) -> str:
        """Move.uci() but without considering side"""
        return sf.render(self.from_square) + sf.render(self.to_square)

    def mirror(self) -> 'Move':
        """
        Returns 'mirrored' move.
        ex) a2a4 -> h7h5
        """
        return Move(119 - self.from_square, 119 - self.to_square)

    def uci(self) -> str:
        """Returns UCI string representation of the move"""
        if self.side: return self.__uci()
        else: return self.mirror().__uci()

    @classmethod
    def from_uci(cls, uci: str) -> 'Move':
        """
        Creates new Move object from given UCI string
        raises ValueError if invalid UCI is given
        """
        match = re.match('([a-h][1-8])'*2, uci)
        if match:
            return Move(sf.parse(match.group(1)), sf.parse(match.group(2)))
        else:
            raise ValueError(f"Invalid uci: {uci}")


class Sunfish:
    def __init__(self):
        self.hist = [sf.Position(sf.initial, 0, (True,True), (True,True), 0, 0)]
        self.searcher = sf.Searcher()

    def turn(self) -> Color:
        return bool(len(self.hist)%2)
        # return self.hist[-1].board.startswith(' ')

    def push(self, move: Move):
        """
        Updates board with given Move object
        raises ValueError if the move is illegal
        """
        import chess
        if move.side!=self.turn():
            move = move.mirror()
        if tuple(move) not in self.hist[-1].gen_moves():
            raise ValueError(f"Illegal move: '{str(move)}'")
        self.hist.append(self.hist[-1].move(move))

    def push_san(self, uci: str):
        """Same as push(Move.from_uci(arg))"""
        self.push(Move.from_uci(uci))

    def pop(self):
        """Undo last move(push)"""
        del self.hist[-1]

    def think(self, timeout: float=1) -> Move:
        """
        Return best Move sunfish can think of within given timeout
        (timeout default set to 1s)
        """
        start = time.time()
        for _depth, move, score in self.searcher.search(self.hist[-1], self.hist):
            if time.time() - start > timeout:
                break
        return Move(*move, self.turn())

    def act(self, timeout: float=1) -> Move:
        """think & push"""
        move = self.think(timeout)
        self.push(move)
        return move

    def __str__(self) -> str:
        return self.ascii()

    def ascii(self) -> str:
        """Returns ascii str representation of board"""
        board = self.hist[-1].board
        if self.turn()==BLACK:
            board = board[::-1].swapcase()
        ret = []
        for row in board.split():
            ret.append(' '.join(c for c in row))
        return '\n'.join(ret)

    def unicode(self) -> str:
        """Returns unicode str representation of board"""
        uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                      'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'·'}
        pos = self.hist[-1] if self.turn()==WHITE else self.hist[-1].rotate()
        ret = []
        for row in pos.board.split():
            ret.append(' '.join(uni_pieces[c] for c in row))
        return '\n'.join(ret)

def play():
    fish = Sunfish()
    while True:
        print(fish.unicode())
        print()
        while True:
            san = input("Your move: ")
            try:
                fish.push_san(san)
            except ValueError:
                print("Please enter a move like g8f6")
            else: break

        print(fish.unicode())
        print()
        move = fish.act()
        print(f"My move: {move.uci()}")

def selfplay():
    fish = Sunfish()
    while True:
        print(fish.unicode())
        print()
        move = fish.act()
        print(move.uci())

if __name__=="__main__":
    options = [play, selfplay]
    print("1. play vs sunfish")
    print("2. watch sunfish vs sunfish")
    chosen = int(input("Your option: ") )
    options[chosen-1]()
