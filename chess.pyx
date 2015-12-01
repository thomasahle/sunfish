#cython: boundscheck=False, wraparound=False

cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython

###############################################################################
# Globals
###############################################################################
cdef enum:
	# Board length
	n = 120

	# Piece constants
	nline = -3
	space = -2
	empty = -1
	opp_pawn = 0
	opp_knight = 1
	opp_bishop = 2
	opp_rook = 3
	opp_queen = 4
	opp_king = 5
	self_pawn = 6
	self_knight = 7
	self_bishop = 8
	self_rook = 9
	self_queen = 10
	self_king = 11

	# Number of piece types
	pieces = 6

	# Our board is represented as a 120 numpy array. The padding allows for
	# fast detection of moves that don't stay within the board.
	A1 = 91
	H1 = 98
	A8 = 21
	H8 = 28

	# Direction constants
	N = -10 
	E = 1
	S = 10
	W = -1

	MAX_DIRS = 8
	MAXINT = 999999

npdirections = np.array([
		   [ N, 2*N, N+W, N+E, MAXINT, MAXINT, MAXINT, MAXINT], # Pawn
		   [ 2*N+E, N+2*E, S+2*E, 2*S+E, 2*S+W, S+2*W, N+2*W, 2*N+W], # Knight
		   [ N+E, S+E, S+W, N+W, MAXINT, MAXINT, MAXINT, MAXINT], # Bishop
		   [ N, E, S, W, MAXINT, MAXINT, MAXINT, MAXINT], # Rook
		   [ N, E, S, W, N+E, S+E, S+W, N+W], # Queen
		   [ N, E, S, W, N+E, S+E, S+W, N+W ]
		], dtype=np.int32) # King

cdef:
	np.int32_t[:, :] directions = npdirections

ctypedef struct Position:
	np.int32_t[:] board
	np.uint8_t[:] wc
	np.uint8_t[:] bc
	np.int32_t ep
	np.int32_t kp
	np.int32_t score

###############################################################################
# Chess logic
###############################################################################

cpdef gen_moves(Position pos):
	cdef:
		int i, j, k
		np.int32_t d, piece, dest
	# For each of our pieces, iterate through each possible 'ray' of moves,
	# as defined in the 'directions' map. The rays are broken e.g. by
	# captures or immediately in case of pieces such as knights.

	result = []

	with nogil:
		for i in range(n):
			piece = pos.board[i]

			# skip if this piece does not belong to player of interest
			if piece < self_pawn:
				continue

			for k in range(MAX_DIRS):
				d = directions[piece % pieces, k]
				if d == MAXINT:
					break

				j = i+d

				while True:
					dest = pos.board[j]

					# Stay inside the board
					if dest == nline or dest == space:
						break

					# Castling
					if i == A1 and dest == self_king and pos.wc[0]:
						with gil:
							result.append((j, j-2))

					if i == H1 and dest == self_king and pos.wc[1]:
						with gil:
							result.append((j, j+2))

					# No friendly captures
					if dest >= self_pawn:
						break

					# Pawn promotion
					if piece == self_pawn and d in (N+W, N+E) and dest == empty and j not in (pos.ep, pos.kp):
						break

					if piece == self_pawn and d in (N, 2*N) and dest != empty:
						break

					if piece == self_pawn and d == 2*N and (i < A1+N or pos.board[i+N] != empty):
						break

					# Move it
					with gil:
						result.append((i, j))

					# Stop crawlers from sliding
					if piece in (self_pawn, self_knight, self_king):
						break

					# No sliding after captures
					if dest >= opp_pawn and dest < self_pawn:
						break

					j += d

	return result

cdef inline void rotate(Position pos) nogil:
	
	cdef int i

	for i in range(n):
		if pos.board[i] >= 0:
			pos.board[i] = (pos.board[i] + 6) % 12

	pos.board[::-1]
	pos.score *= -1
	pos.ep = 119-pos.ep
	pos.kp = 119-pos.kp

cpdef Position make_move(Position pos, np.int32_t[:] move):
	cdef:
		np.int32_t i, j, piece, dest
		Position new_pos

	# Grab source and destination of move
	i = move[0]
	j = move[1]

	piece = pos.board[i]
	dest = pos.board[j]

	# Create copy of variables and apply move
	new_pos.board = pos.board.copy()
	new_pos.wc = pos.wc.copy()
	new_pos.bc = pos.bc.copy()
	new_pos.ep = 0
	new_pos.kp = 0

	new_pos.board[j] = pos.board[i]
	new_pos.board[i] = empty

	# Castling rights
	if i == A1:
		new_pos.wc[0] = 0
		new_pos.wc[1] = pos.wc[1]

	if i == H1:
		new_pos.wc[0] = pos.wc[0]
		new_pos.wc[1] = 0

	if j == A8:
		new_pos.bc[0] = pos.bc[0]
		new_pos.bc[1] = 0

	if j == H8:
		new_pos.wc[0] = 0
		new_pos.bc[1] = pos.bc[1]

	# Castling
	if piece == self_king:
		new_pos.wc[0] = 0
		new_pos.wc[1] = 0
		if abs(j-i) == 2:
			new_pos.kp = (i+j)//2
			new_pos.board[A1 if j < i else H1] = empty
			new_pos.board[new_pos.kp] = self_rook

	# Pawn promotion
	if piece == self_pawn:
		if A8 <= j and j <= H8:
			new_pos.board[j] = self_queen
		if j - i == 2*N:
			ep = i + N
		if j - i in (N+W, N+E) and dest == empty:
			new_pos.board[j+S] = empty

	# Return result
	# new_pos.score = score + evaluation(new_pos.board)
	rotate(new_pos)
	return new_pos





