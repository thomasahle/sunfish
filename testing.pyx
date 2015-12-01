#cython: boundscheck=False,
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from cpython cimport bool

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT

cpdef void testing(np.int32_t[:, :] board) nogil:
	with gil:
		print(board)
		print("ayy")
		print("HEYY")


# mapping = { '\n': -3, ' ': -2, '.': -1, 
#            'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
#            'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11 }

cdef:
	# 20,000 cutoff value derived by Claude Shannon
	np.int32_t[:] piece_vals = np.array([
		100, 320, 330, 500, 900, 20000], dtype=np.int32
	)

	np.int32_t[:, :] pst_vals = \
		np.ndarray(shape=(6,8), dtype=np.int32, buffer=np.array([
			# Pawn
			[0,  0,  0,  0,  0,  0,  0,  0,
			50, 50, 50, 50, 50, 50, 50, 50,
			10, 10, 20, 30, 30, 20, 10, 10,
			 5,  5, 10, 25, 25, 10,  5,  5,
			 0,  0,  0, 20, 20,  0,  0,  0,
			 5, -5,-10,  0,  0,-10, -5,  5,
			 5, 10, 10,-20,-20, 10, 10,  5,
			 0,  0,  0,  0,  0,  0,  0,  0],
			# Knight
			[-50,-40,-30,-30,-30,-30,-40,-50,
			-40,-20,  0,  0,  0,  0,-20,-40,
			-30,  0, 10, 15, 15, 10,  0,-30,
			-30,  5, 15, 20, 20, 15,  5,-30,
			-30,  0, 15, 20, 20, 15,  0,-30,
			-30,  5, 10, 15, 15, 10,  5,-30,
			-40,-20,  0,  5,  5,  0,-20,-40,
			-50,-40,-30,-30,-30,-30,-40,-50],
			# Bishop
			[-20,-10,-10,-10,-10,-10,-10,-20,
			-10,  0,  0,  0,  0,  0,  0,-10,
			-10,  0,  5, 10, 10,  5,  0,-10,
			-10,  5,  5, 10, 10,  5,  5,-10,
			-10,  0, 10, 10, 10, 10,  0,-10,
			-10, 10, 10, 10, 10, 10, 10,-10,
			-10,  5,  0,  0,  0,  0,  5,-10,
			-20,-10,-10,-10,-10,-10,-10,-20],
			# Rook
			[ 0,  0,  0,  0,  0,  0,  0,  0,
			  5, 10, 10, 10, 10, 10, 10,  5,
			 -5,  0,  0,  0,  0,  0,  0, -5,
			 -5,  0,  0,  0,  0,  0,  0, -5,
			 -5,  0,  0,  0,  0,  0,  0, -5,
			 -5,  0,  0,  0,  0,  0,  0, -5,
			 -5,  0,  0,  0,  0,  0,  0, -5,
			  0,  0,  0,  5,  5,  0,  0,  0],
			# Queen
			[-20,-10,-10, -5, -5,-10,-10,-20,
			-10,  0,  0,  0,  0,  0,  0,-10,
			-10,  0,  5,  5,  5,  5,  0,-10,
			 -5,  0,  5,  5,  5,  5,  0, -5,
			  0,  0,  5,  5,  5,  5,  0, -5,
			-10,  5,  5,  5,  5,  5,  0,-10,
			-10,  0,  5,  0,  0,  0,  0,-10,
			-20,-10,-10, -5, -5,-10,-10,-20],
			# King
			[-30,-40,-40,-50,-50,-40,-40,-30,
			-30,-40,-40,-50,-50,-40,-40,-30,
			-30,-40,-40,-50,-50,-40,-40,-30,
			-30,-40,-40,-50,-50,-40,-40,-30,
			-20,-30,-30,-40,-40,-30,-30,-20,
			-10,-20,-20,-20,-20,-20,-20,-10,
			 20, 20,  0,  0,  0,  0, 20, 20,
			 20, 30, 10,  0,  0, 10, 30, 20]
			])
		)

	# Endgame
	np.int32_t[:] king_end = np.array(
		[-50,-40,-30,-20,-20,-30,-40,-50,
		-30,-20,-10,  0,  0,-10,-20,-30,
		-30,-10, 20, 30, 30, 20,-10,-30,
		-30,-10, 30, 40, 40, 30,-10,-30,
		-30,-10, 30, 40, 40, 30,-10,-30,
		-30,-10, 20, 30, 30, 20,-10,-30,
		-30,-30,  0,  0,  0,  0,-30,-30,
		-50,-30,-30,-30,-30,-30,-30,-50],
		dtype = np.int32
	)

	np.int32_t[:] pawn_end = np.array(
		[0,  0,  0,  0,  0,  0,  0,  0,
		50, 50, 50, 50, 50, 50, 50, 50,
		30, 30, 30, 30, 30, 30, 30, 30,
		20, 20, 20, 25, 25, 20, 20, 20,
		10, 10, 10, 15, 15, 10, 10, 10,
		 0,  0,  0,  0,  0,  0,  0,  0,
	   -15,-20,-20,-20,-20,-20,-20,-15,
		 0,  0,  0,  0,  0,  0,  0,  0],
		dtype = np.int32
	)

cdef np.int32_t total_material(np.int32_t[:] board) nogil:
	cdef:
		np.int32_t amt = 0
		np.int32_t piece

	for idx in range(120):
		piece = board[idx]
		if piece >= 0:
			amt += piece_vals[piece % 6]

	return amt


cdef np.int32_t is_endgame(np.int32_t[:] board) nogil:
	cdef np.int32_t ret_val = 1
	# material cutoff
	# roughly 2 Kings, 2 Rooks, 1 Minor, 6 Pawns each
	if total_material(board) > 44000:
		ret_val = 0

	return ret_val


cpdef np.int32_t evaluate(np.int32_t[:] board) nogil:
	cdef:
		np.int32_t score = 0
		np.int32_t row, col, pos, piece, endgame_bool, idx

	endgame_bool = is_endgame(board)
	for idx in range(120):
		piece = board[idx]
		
		if piece >= 0:
			row = idx / 10 - 2
			col = idx % 10 - 1
			pos = row * 10 + col

			# My piece
			if piece <= 5:
				score += piece_vals[piece]

				if endgame_bool == 1:
					if piece == 0: score += pawn_end[pos]
					elif piece == 5: score += king_end[pos]
					else: score += pst_vals[piece][pos]
				else:
					score += pst_vals[piece][pos]

			else:
				score -= piece_vals[piece % 6]

				if endgame_bool == 1:
					if piece == 6: score -= pawn_end[pos]
					elif piece == 11: score -= king_end[pos]
					else: score -= pst_vals[piece % 6][pos]
				else:
					score -= pst_vals[piece % 6][pos]

	return score







