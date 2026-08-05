#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

/*
 * Translation of Sunfish Python code to C.
 * This code is a single-file C version of the chess engine logic.
 * It is not optimized and may be slower than the original Python (which used PyPy).
 * Some simplifications have been made. The logic closely follows the original code.
 */

#define VERSION "sunfish 2023"

/* Piece values */
static const int piece_score[6] = {
    100,   /* P */
    280,   /* N */
    320,   /* B */
    479,   /* R */
    929,   /* Q */
    60000  /* K */
};

/* Map piece char to index: P=0,N=1,B=2,R=3,Q=4,K=5 */
static inline int piece_index(char c) {
    switch (c) {
        case 'P': return 0;
        case 'N': return 1;
        case 'B': return 2;
        case 'R': return 3;
        case 'Q': return 4;
        case 'K': return 5;
        default: return -1;
    }
}

/* Board indexing and directions */
#define N   (-10)
#define E   1
#define S   10
#define W   (-1)
#define A1  91
#define H1  98
#define A8  21
#define H8  28

/* Mating constants */
#define MATE_LOWER (60000 - 10 * 929)
#define MATE_UPPER (60000 + 10 * 929)

/* Search tuning constants */
static int QS = 40;
static int QS_A = 140;
static int EVAL_ROUGHNESS = 15;

/* Directions for pieces */
static const int directions_P[4] = {N, N+N, N+W, N+E};
static const int directions_N[8] = {N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W};
static const int directions_B[4] = {N+E, S+E, S+W, N+W};
static const int directions_R[4] = {N, E, S, W};
static const int directions_Q[8] = {N, E, S, W, N+E, S+E, S+W, N+W};
static const int directions_K[8] = {N, E, S, W, N+E, S+E, S+W, N+W};

/* We'll store PST as a large table indexed by piece and position. 
   pst[<piece>][<square>] gives the score contribution. */
static int pst[6][120]; 
/* We will define the base piece-square tables as given in Python code, then adjust. */

/* Raw PST data as in Python code, indexed by piece in order P,N,B,R,Q,K (0-based) */
static const int base_pst[6][64] = {
    /* P */
    {
       0,   0,   0,   0,   0,   0,   0,   0,
      78,  83,  86,  73, 102,  82,  85,  90,
       7,  29,  21,  44,  40,  31,  44,   7,
     -17,  16,  -2,  15,  14,   0,  15, -13,
     -26,   3,  10,   9,   6,   1,   0, -23,
     -22,   9,   5, -11, -10,  -2,   3, -19,
     -31,   8,  -7, -37, -36, -14,   3, -31,
       0,   0,   0,   0,   0,   0,   0,   0
    },
    /* N */
    {
     -66, -53, -75, -75, -10, -55, -58, -70,
      -3,  -6, 100, -36,   4,  62,  -4, -14,
      10,  67,   1,  74,  73,  27,  62,  -2,
      24,  24,  45,  37,  33,  41,  25,  17,
      -1,   5,  31,  21,  22,  35,   2,   0,
     -18,  10,  13,  22,  18,  15,  11, -14,
     -23, -15,   2,   0,   2,   0, -23, -20,
     -74, -23, -26, -24, -19, -35, -22, -69
    },
    /* B */
    {
     -59, -78, -82, -76, -23,-107, -37, -50,
     -11,  20,  35, -42, -39,  31,   2, -22,
      -9,  39, -32,  41,  52, -10,  28, -14,
      25,  17,  20,  34,  26,  25,  15,  10,
      13,  10,  17,  23,  17,  16,   0,   7,
      14,  25,  24,  15,   8,  25,  20,  15,
      19,  20,  11,   6,   7,   6,  20,  16,
      -7,   2, -15, -12, -14, -15, -10, -10
    },
    /* R */
    {
      35,  29,  33,   4,  37,  33,  56,  50,
      55,  29,  56,  67,  55,  62,  34,  60,
      19,  35,  28,  33,  45,  27,  25,  15,
       0,   5,  16,  13,  18,  -4,  -9,  -6,
     -28, -35, -16, -21, -13, -29, -46, -30,
     -42, -28, -42, -25, -25, -35, -26, -46,
     -53, -38, -31, -26, -29, -43, -44, -53,
     -30, -24, -18,   5,  -2, -18, -31, -32
    },
    /* Q */
    {
       6,   1,  -8,-104,  69,  24,  88,  26,
      14,  32,  60, -10,  20,  76,  57,  24,
      -2,  43,  32,  60,  72,  63,  43,   2,
       1, -16,  22,  17,  25,  20, -13,  -6,
     -14, -15,  -2,  -5,  -1, -10, -20, -22,
     -30,  -6, -13, -11, -16, -11, -16, -27,
     -36, -18,   0, -19, -15, -15, -21, -38,
     -39, -30, -31, -13, -31, -36, -34, -42
    },
    /* K */
    {
       4,  54,  47, -99, -99,  60,  83, -62,
     -32,  10,  55,  56,  56,  55,  10,   3,
     -62,  12, -57,  44, -67,  28,  37, -31,
     -55,  50,  11,  -4, -19,  13,   0, -49,
     -55, -43, -52, -28, -51, -47,  -8, -50,
     -47, -42, -43, -79, -64, -32, -29, -32,
      -4,   3, -14, -50, -57, -18,  13,   4,
      17,  30,  -3, -14,   6,  -1,  40,  18
    }
};

/* initial board */
static const char *initial =
"         \n"
"         \n"
" rnbqkbnr\n"
" pppppppp\n"
" ........\n"
" ........\n"
" ........\n"
" ........\n"
" PPPPPPPP\n"
" RNBQKBNR\n"
"         \n"
"         \n";

typedef struct {
    int i, j;
    char prom;
} Move;

typedef struct {
    /* board: 120-char representation */
    char board[121];
    int score;
    bool wc[2]; /* White castling rights [queen-side, king-side] */
    bool bc[2]; /* Black castling rights [queen-side, king-side] */
    int ep;
    int kp;
} Position;

/* Transposition table entries */
typedef struct {
    int lower, upper;
} Entry;

typedef struct {
    /* Key: (pos, depth, can_null) is complicated to store. We skip hashing and do a simplistic approach.
       A real engine would need a hash. For simplicity, we won't fully implement a large TT here.
       We'll implement a very naive small TT. */
    /* WARNING: This is a simplistic replacement. In a serious engine, you'd use Zobrist keys. */

    /* We'll store a small fixed-size table. On collisions we overwrite. */
    struct {
        Position pos;
        int depth;
        bool can_null;
        Entry entry;
        bool used;
    } tp_score[10000];

    struct {
        Position pos;
        Move move;
        bool used;
    } tp_move[10000];

    int nodes;
    Position history[1024];
    int hist_len;
} Searcher;

/* Very naive hashing for storing pos in table */
static unsigned pos_hash(const Position *p) {
    /* We'll just hash the board. This is weak but easier for demonstration. */
    unsigned h = 0;
    for (int i=0; i<120; i++) {
        h = h * 31 + (unsigned char)p->board[i];
    }
    h = h * 31 + (unsigned)(p->score & 0xFFFF);
    return h % 10000;
}

static bool pos_equal(const Position *a, const Position *b) {
    if (a->score != b->score) return false;
    if (a->ep != b->ep) return false;
    if (a->kp != b->kp) return false;
    if (a->wc[0] != b->wc[0] || a->wc[1] != b->wc[1]) return false;
    if (a->bc[0] != b->bc[0] || a->bc[1] != b->bc[1]) return false;
    return (memcmp(a->board, b->board, 120) == 0);
}

static void tp_score_put(Searcher *searcher, const Position *pos, int depth, bool can_null, Entry entry) {
    unsigned h = pos_hash(pos);
    for (int i=0; i<100; i++) {
        unsigned idx = (h + i) % 10000;
        if (!searcher->tp_score[idx].used ||
            pos_equal(&searcher->tp_score[idx].pos, pos)) {
            searcher->tp_score[idx].pos = *pos;
            searcher->tp_score[idx].depth = depth;
            searcher->tp_score[idx].can_null = can_null;
            searcher->tp_score[idx].entry = entry;
            searcher->tp_score[idx].used = true;
            return;
        }
    }
}

static bool tp_score_get(Searcher *searcher, const Position *pos, int depth, bool can_null, Entry *entry) {
    unsigned h = pos_hash(pos);
    for (int i=0; i<100; i++) {
        unsigned idx = (h + i) % 10000;
        if (searcher->tp_score[idx].used &&
            searcher->tp_score[idx].depth == depth &&
            searcher->tp_score[idx].can_null == can_null &&
            pos_equal(&searcher->tp_score[idx].pos, pos)) {
            *entry = searcher->tp_score[idx].entry;
            return true;
        }
    }
    return false;
}

static void tp_move_put(Searcher *searcher, const Position *pos, Move move) {
    unsigned h = pos_hash(pos);
    for (int i=0; i<100; i++) {
        unsigned idx = (h + i) % 10000;
        if (!searcher->tp_move[idx].used ||
            pos_equal(&searcher->tp_move[idx].pos, pos)) {
            searcher->tp_move[idx].pos = *pos;
            searcher->tp_move[idx].move = move;
            searcher->tp_move[idx].used = true;
            return;
        }
    }
}

static bool tp_move_get(Searcher *searcher, const Position *pos, Move *move) {
    unsigned h = pos_hash(pos);
    for (int i=0; i<100; i++) {
        unsigned idx = (h + i) % 10000;
        if (searcher->tp_move[idx].used &&
            pos_equal(&searcher->tp_move[idx].pos, pos)) {
            *move = searcher->tp_move[idx].move;
            return true;
        }
    }
    return false;
}

/* Rotate a position */
static Position rotate(const Position *pos, bool nullmove) {
    Position r = *pos;
    // Reverse board
    for (int i=0; i<120; i++) {
        char c = pos->board[119 - i];
        if (isalpha((unsigned char)c)) {
            if (isupper((unsigned char)c)) c = (char)tolower((unsigned char)c);
            else c = (char)toupper((unsigned char)c);
        }
        r.board[i] = c;
    }
    r.score = -pos->score;
    r.wc[0] = pos->bc[0];
    r.wc[1] = pos->bc[1];
    r.bc[0] = pos->wc[0];
    r.bc[1] = pos->wc[1];
    if (!nullmove && pos->ep) r.ep = 119 - pos->ep; else r.ep = 0;
    if (!nullmove && pos->kp) r.kp = 119 - pos->kp; else r.kp = 0;
    return r;
}

static void put_char(char *board, int i, char p) {
    board[i] = p;
}

/* Compute value of a move */
static int position_value(const Position *pos, Move move);

/* Make a move and return new position */
static Position position_move(const Position *pos, Move move) {
    int i = move.i, j = move.j;
    char prom = move.prom;
    char p = pos->board[i];
    char q = pos->board[j];

    Position npos = *pos;
    int score = pos->score + position_value(pos, move);
    put_char(npos.board, j, p);
    put_char(npos.board, i, '.');

    bool wc0 = npos.wc[0], wc1 = npos.wc[1];
    bool bc0 = npos.bc[0], bc1 = npos.bc[1];
    int ep = 0, kp = 0;

    if (i == A1) wc0 = false;
    if (i == H1) wc1 = false;
    if (j == A8) bc1 = false;
    if (j == H8) bc0 = false;

    if (p == 'K') {
        wc0 = wc1 = false;
        if (abs(j - i) == 2) {
            kp = (i + j) / 2;
            if (j < i) {
                /* Queen side castling */
                put_char(npos.board, A1, '.');
                put_char(npos.board, kp, 'R');
            } else {
                /* King side castling */
                put_char(npos.board, H1, '.');
                put_char(npos.board, kp, 'R');
            }
        }
    }

    if (p == 'P') {
        if (j >= A8 && j <= H8) {
            put_char(npos.board, j, prom);
        }
        if (j - i == 2*N) {
            ep = i + N;
        }
        if (j == pos->ep) {
            put_char(npos.board, j+S, '.');
        }
    }

    npos.score = score;
    npos.wc[0] = wc0; npos.wc[1] = wc1;
    npos.bc[0] = bc0; npos.bc[1] = bc1;
    npos.ep = ep;
    npos.kp = kp;

    /* rotate */
    npos = rotate(&npos, false);
    return npos;
}

static int position_value(const Position *pos, Move move) {
    int i = move.i, j = move.j;
    char p = pos->board[i];
    char q = pos->board[j];
    int score = pst[piece_index((char)toupper((unsigned char)p))][j] - pst[piece_index((char)toupper((unsigned char)p))][i];
    if (q != '.' && !isspace((unsigned char)q)) {
        score += pst[piece_index((char)toupper((unsigned char)q))][119 - j];
    }
    if (abs(j - pos->kp) < 2 && pos->kp) {
        score += pst[piece_index('K')][119 - j];
    }
    if (p == 'K' && abs(i - j) == 2) {
        score += pst[piece_index('R')][(i+j)/2];
        score -= pst[piece_index('R')][ (j<i)? A1:H1 ];
    }
    if (p == 'P') {
        if (j >= A8 && j <= H8) {
            int pidx = piece_index('P');
            int pprom = piece_index((char)toupper((unsigned char)move.prom));
            score += pst[pprom][j] - pst[pidx][j];
        }
        if (j == pos->ep) {
            score += pst[piece_index('P')][119-(j+S)];
        }
    }
    return score;
}

/* Generate moves */
typedef struct {
    Move moves[256];
    int count;
} MoveList;

static bool is_upper(char c) {
    return (c >= 'A' && c <= 'Z');
}

static bool is_lower(char c) {
    return (c >= 'a' && c <= 'z');
}

static bool is_space(char c) {
    return (c == ' ' || c == '\n' || c == '\t');
}

static const int *piece_dirs(char p, int *count) {
    switch (p) {
        case 'P': *count=4; return directions_P;
        case 'N': *count=8; return directions_N;
        case 'B': *count=4; return directions_B;
        case 'R': *count=4; return directions_R;
        case 'Q': *count=8; return directions_Q;
        case 'K': *count=8; return directions_K;
        default: *count=0; return NULL;
    }
}

static void gen_moves(const Position *pos, MoveList *mlist) {
    mlist->count = 0;
    for (int i=0; i<120; i++) {
        char p = pos->board[i];
        if (!is_upper(p)) continue;
        int dcount;
        const int *dirs = piece_dirs(p, &dcount);
        for (int di=0; di<dcount; di++) {
            int d = dirs[di];
            for (int j=i+d; ; j+=d) {
                char q = pos->board[j];
                if (is_space(q) || is_upper(q)) break;
                if (p == 'P') {
                    if ((d == N || d == N+N) && q != '.') break;
                    if (d == N+N && (i < A1+N || pos->board[i+N] != '.')) break;
                    if ((d == N+W || d == N+E) && q == '.' && j != pos->ep && j != pos->kp && j != pos->kp-1 && j != pos->kp+1) break;
                    if (j >= A8 && j <= H8) {
                        const char *proms = "NBRQ";
                        for (int pi=0; pi<4; pi++) {
                            Move m = {i,j,proms[pi]};
                            mlist->moves[mlist->count++] = m;
                        }
                        break;
                    }
                }

                Move m = {i,j,0};
                m.prom = 0;
                mlist->moves[mlist->count++] = m;

                if (p=='P' || p=='N' || p=='K' || is_lower(q)) break;
                if (i==A1 && pos->board[j+E]=='K' && pos->wc[0]) {
                    Move m2 = {j+E, j+W,0};
                    mlist->moves[mlist->count++] = m2;
                }
                if (i==H1 && pos->board[j+W]=='K' && pos->wc[1]) {
                    Move m2 = {j+W, j+E,0};
                    mlist->moves[mlist->count++] = m2;
                }
            }
        }
    }
}

/* Check repetition in history */
static bool position_in_history(Searcher *searcher, const Position *pos) {
    for (int i=0; i<searcher->hist_len; i++) {
        if (pos_equal(&searcher->history[i], pos)) return true;
    }
    return false;
}

/* Search logic */

static int bound(Searcher *searcher, Position pos, int gamma, int depth, bool can_null);

static int position_bound(Searcher *searcher, Position pos, int gamma, int depth, bool can_null) {
    return bound(searcher, pos, gamma, depth, can_null);
}

static int bound(Searcher *searcher, Position pos, int gamma, int depth, bool can_null) {
    searcher->nodes++;

    depth = depth>0?depth:0;

    if (pos.score <= -MATE_LOWER) {
        return -MATE_UPPER;
    }

    Entry entry = { -MATE_UPPER, MATE_UPPER };
    if (tp_score_get(searcher, &pos, depth, can_null, &entry)) {
        if (entry.lower >= gamma) return entry.lower;
        if (entry.upper < gamma) return entry.upper;
    }

    if (can_null && depth > 0 && position_in_history(searcher, &pos)) {
        return 0;
    }

    /* Moves generator */
    /* We'll implement a local function to get moves and evaluate inline. */
    /* We do a form of MTD-bi search and futility pruning etc. */

    /* Let's try null-move */
    int best = -MATE_UPPER;

    if (depth > 2 && can_null && abs(pos.score) < 500) {
        Position nullpos = rotate(&pos, true);
        int score_null = -position_bound(searcher, nullpos, 1 - gamma, depth - 3, false);
        if (score_null > best) {
            best = score_null;
            if (best >= gamma) {
                /* Save entry */
                Entry new_entry = {best, entry.upper};
                tp_score_put(searcher, &pos, depth, can_null, new_entry);
                return best;
            }
        }
    }

    if (depth == 0) {
        /* QSearch stand pat */
        if (pos.score > best) best = pos.score;
        if (best >= gamma) {
            Entry new_entry = {best, entry.upper};
            tp_score_put(searcher, &pos, depth, can_null, new_entry);
            return best;
        }
    }

    /* Killer move */
    Move killer_move; 
    bool has_killer = tp_move_get(searcher, &pos, &killer_move);

    int val_lower = QS - depth * QS_A;

    /* If we have a killer move, try it */
    if (has_killer && position_value(&pos, killer_move) >= val_lower) {
        Position newpos = position_move(&pos, killer_move);
        int sc = -position_bound(searcher, newpos, 1 - gamma, depth - 1, true);
        if (sc > best) best = sc;
        if (best >= gamma) {
            tp_move_put(searcher, &pos, killer_move);
            Entry new_entry = {best, entry.upper};
            tp_score_put(searcher, &pos, depth, can_null, new_entry);
            return best;
        }
    }

    MoveList mlist;
    gen_moves(&pos, &mlist);
    /* Sort by value */
    int vals[256];
    for (int mi=0; mi<mlist.count; mi++) {
        vals[mi] = position_value(&pos, mlist.moves[mi]);
    }
    /* sort by vals descending */
    for (int x=0; x<mlist.count-1; x++) {
        for (int y=x+1; y<mlist.count; y++) {
            if (vals[y] > vals[x]) {
                int tv = vals[x]; vals[x]=vals[y]; vals[y]=tv;
                Move tm = mlist.moves[x]; mlist.moves[x]=mlist.moves[y]; mlist.moves[y]=tm;
            }
        }
    }

    for (int mi=0; mi<mlist.count; mi++) {
        Move m = mlist.moves[mi];
        int val = vals[mi];

        if (depth <= 1 && pos.score + val < gamma) {
            int sc = (val < MATE_LOWER) ? pos.score + val : MATE_UPPER;
            if (sc > best) best = sc;
            if (best >= gamma) {
                tp_move_put(searcher, &pos, m);
                Entry new_entry = {best, entry.upper};
                tp_score_put(searcher, &pos, depth, can_null, new_entry);
                return best;
            }
            break;
        }

        Position newpos = position_move(&pos, m);
        int sc = -position_bound(searcher, newpos, 1 - gamma, depth - 1, true);
        if (sc > best) best = sc;
        if (best >= gamma) {
            tp_move_put(searcher, &pos, m);
            Entry new_entry = {best, entry.upper};
            tp_score_put(searcher, &pos, depth, can_null, new_entry);
            return best;
        }
    }

    if (depth > 2 && best == -MATE_UPPER) {
        Position flipped = rotate(&pos, true);
        int in_check = (position_bound(searcher, flipped, MATE_UPPER, 0, true) == MATE_UPPER);
        best = in_check ? -MATE_LOWER : 0;
    }

    if (best >= gamma) {
        Entry new_entry = {best, entry.upper};
        tp_score_put(searcher, &pos, depth, can_null, new_entry);
    }
    if (best < gamma) {
        Entry new_entry = {entry.lower, best};
        tp_score_put(searcher, &pos, depth, can_null, new_entry);
    }

    return best;
}

/* Iterative deepening search */
static Move search(Searcher *searcher) {
    /* We'll do a truncated version of the iterative deepening and MTD-bi logic. */
    Move best_move = {0,0,0};
    Position root = searcher->history[searcher->hist_len - 1];

    int gamma = 0;
    time_t start = time(NULL);
    double think = 1.0; /* Fixed think time for demo */

    for (int depth=1; depth<50; depth++) {
        int lower = -MATE_LOWER;
        int upper = MATE_LOWER;
        while (lower < upper - EVAL_ROUGHNESS) {
            int score = position_bound(searcher, root, gamma, depth, false);
            if (score >= gamma) {
                lower = score;
            } else {
                upper = score;
            }

            /* Check tp_move for best_move */
            Move mv;
            if (tp_move_get(searcher, &root, &mv) && score >= gamma) {
                best_move = mv;
            }

            gamma = (lower + upper + 1) / 2;

            if (difftime(time(NULL), start) > think*0.8) {
                return best_move;
            }
        }

        if (difftime(time(NULL), start) > think*0.8) {
            break;
        }
    }

    return best_move;
}

/* Helper functions for parsing/moving */

static int parse_sq(const char *c) {
    int fil = c[0]-'a';
    int rank = c[1]-'1';
    return A1 + fil - 10*rank;
}

static void render_sq(int i, char *out) {
   printf("i=%d\n", i);
   printf("f=%d\n", (i - A1 + 120) % 10);
   printf("r=%d\n", (A1 - i + 9) / 10);
    out[0] = (char)('a' + (i - A1 + 120) % 10);
    out[1] = (char)('1' + (A1 - i + 9) / 10);
    out[2] = '\0';
}


int main() {
    setbuf(stdout, NULL);
    /* Initialize PST */
    /* We must pad as done in python code:
       The python code pads 20 empty lines top and bottom and also shifts piece values.
       We'll mimic the final indexing: 
       The board indexing is already 120. We'll place PST accordingly:
       For each piece: pad with 20 zeros at start and 20 at end, and add piece value. 
    */
    int piece_val[6];
    for (int i=0; i<6; i++)
       piece_val[i] = piece_score[i];

    for (int pc=0; pc<6; pc++) {
        // base_pst[pc] is an 8x8 block
        // We'll create a 120 char array with padding
        // According to code: 2 rows of (0)*10 above and below = 20 zeros top and bottom
        for (int idx=0; idx<120; idx++)
           pst[pc][idx] = 0;
        // Insert the 8x8 block at rows 2..9 in board indexing
        // The board indexing lines for actual board: line 2..9 (20..99)
        for (int r=0; r<8; r++)
            for (int f=0; f<8; f++)
                pst[pc][20 + r*10 + f] = base_pst[pc][r*8+f] + piece_val[pc];
    }

    /* Initialize position */
    Position startpos;
    memset(&startpos, 0, sizeof(startpos));
    strncpy(startpos.board, initial, 120);
    startpos.board[120] = '\0';
    startpos.score = 0;
    startpos.wc[0] = true; startpos.wc[1] = true;
    startpos.bc[0] = true; startpos.bc[1] = true;
    startpos.ep = 0;
    startpos.kp = 0;

    Searcher searcher;
    memset(&searcher, 0, sizeof(searcher));
    searcher.history[0] = startpos;
    searcher.hist_len = 1;

    /* A simple REPL-like loop with minimal UCI support */
    char line[256];
    while (1) {
        if (!fgets(line, sizeof(line), stdin)) break;
        char *cmd = strtok(line, " \n\t");
        if (!cmd) continue;
        if (strcmp(cmd, "uci")==0) {
            printf("id name %s\n", VERSION);
            printf("uciok\n");
        } else if (strcmp(cmd, "isready")==0) {
            printf("readyok\n");
        } else if (strcmp(cmd, "quit")==0) {
            break;
        } else if (strcmp(cmd, "position")==0) {
            char *arg = strtok(NULL, " \n\t");
            if (arg && strcmp(arg,"startpos")==0) {
                /* reset history */
                searcher.hist_len = 1;
                searcher.history[0] = startpos;
                char *next = strtok(NULL, " \n\t");
                if (next && strcmp(next,"moves")==0) {
                    /* apply moves */
                    char *mstr;
                    int ply=0;
                    while ((mstr = strtok(NULL, " \n\t"))) {
                        int i = parse_sq(mstr);
                        int j = parse_sq(mstr+2);
                        char prom = '\0';
                        if (strlen(mstr)>4) {
                            prom = toupper((unsigned char)mstr[4]);
                        } else prom = '\0';
                        Move m = {i,j,prom};
                        if (ply % 2 == 1) {
                            m.i = 119 - m.i;
                            m.j = 119 - m.j;
                        }
                        Position np = position_move(&searcher.history[searcher.hist_len-1], m);
                        searcher.history[searcher.hist_len++] = np;
                        ply++;
                    }
                }
            }
        } else if (strcmp(cmd,"go")==0) {
            /* For simplicity, ignore time and do a short search */
            Move bm = search(&searcher);
            int i = bm.i;
            int j = bm.j;
            if ((searcher.hist_len % 2)==0) {
                i = 119 - i;
                j = 119 - j;
            }
            char move_str[6];
            render_sq(i, move_str);
            render_sq(j, move_str+2);
            if (bm.prom) {
                move_str[4] = (char)tolower((unsigned char)bm.prom);
                move_str[5] = '\0';
            } else {
                move_str[4]='\0';
            }
            printf("bestmove %s\n", move_str);
        }
    }

    return 0;
}

