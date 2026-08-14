/* sunfish.c -- a node-identical C twin of classic sunfish.py (repo root).
 *
 * LAB INSTRUMENT, NOT AN ENGINE RELEASE.  Its single purpose is to make
 * fixed-node tuning games cheap while provably searching the exact same
 * tree as the Python reference: same position in => same chosen move, same
 * node count, same score out.  tools/ctwin/difftest.py measures that claim
 * continuously; see README.md in this directory for the fidelity contract.
 *
 * Design rules (why the code looks the way it does):
 *  - The 120-char string board, the direction constants, the generator
 *    phase order and the consumer loop of bound() are TRANSCRIBED from
 *    sunfish.py, not redesigned.  Speed comes from C, not from a rewrite.
 *  - Python floor semantics: every division that can see a negative
 *    operand goes through pyfloordiv/pymod.
 *  - Python dict semantics: hash maps keep insertion order; updates keep
 *    the original slot; FIFO eviction removes the oldest entry, and the
 *    killer-table eviction skips the search root, like sunfish.py.
 *  - sorted(..., reverse=True) on (val, Move) tuples: keys are unique, so
 *    the order is total -- descending (val, i, j, prom) with prom compared
 *    as a byte ('\0' < 'B' < 'N' < 'Q' < 'R'), exactly Python's tuple order.
 *  - All eval data is injected from a table file (gen_tables.py) and all
 *    search constants are runtime knobs ("set NAME VALUE" or SF_<NAME> env),
 *    so tuning needs no recompilation.
 *
 * Reference: sunfish.py at the repo root on branch nnue-4k (capped null
 * move, mate-distance scoring, IID at depth > 3).  Master-flavor deltas are
 * reachable by knob: set IID_MIN_DEPTH 2, set MATE_DIST 0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <setjmp.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Board geometry (must match sunfish.py exactly)                      */
/* ------------------------------------------------------------------ */
enum { A1 = 91, H1 = 98, A8 = 21, H8 = 28 };
enum { Nd = -10, Ed = 1, Sd = 10, Wd = -1 };

static const int DIR_P[] = { Nd, Nd + Nd, Nd + Wd, Nd + Ed };
static const int DIR_N[] = { Nd+Nd+Ed, Ed+Nd+Ed, Ed+Sd+Ed, Sd+Sd+Ed,
                             Sd+Sd+Wd, Wd+Sd+Wd, Wd+Nd+Wd, Nd+Nd+Wd };
static const int DIR_B[] = { Nd+Ed, Sd+Ed, Sd+Wd, Nd+Wd };
static const int DIR_R[] = { Nd, Ed, Sd, Wd };
static const int DIR_Q[] = { Nd, Ed, Sd, Wd, Nd+Ed, Sd+Ed, Sd+Wd, Nd+Wd };

static const char INITIAL[121] =
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

/* ------------------------------------------------------------------ */
/* Injected evaluation data                                            */
/* ------------------------------------------------------------------ */
static int TAB[6][120];          /* padded pst for P N B R Q K(mid) */
static int KEND[120];            /* endgame king table */
static int PIECEVAL[6];          /* bare piece values P N B R Q K */
static int *PSTP[128];           /* by piece char; PSTP['K'] is swapped */
static int MATE_LOWER, MATE_UPPER;
static int tables_loaded = 0;

/* Runtime knobs.  Defaults reproduce sunfish.py at the repo root. */
static int QS = 40;
static int QS_A = 140;
static int EVAL_ROUGHNESS = 15;
static long TABLE_SIZE = 1000000;
static int NULL_MARGIN = -1;     /* -1: track EVAL_ROUGHNESS (classic ties them) */
static int NULL_MIN_DEPTH = 2;   /* null move when depth > this */
static int NULL_LIMIT = 500;     /* |score| bound for trying null */
static int NULL_RED = 3;         /* null move depth reduction */
static int IID_MIN_DEPTH = 3;    /* IID when depth > this (master: 2) */
static int IID_RED = 3;          /* IID depth reduction */
static int FUT_MAX = 1;          /* futility pruning when depth <= this */
static int MATE_DIST = 1;        /* mate scores carry distance (master: 0) */

/* ------------------------------------------------------------------ */
/* Python arithmetic                                                   */
/* ------------------------------------------------------------------ */
static int pyfloordiv(int a, int b) {
    int q = a / b, r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q--;
    return q;
}
static int pymod(int a, int b) {
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
static int iabs(int x) { return x < 0 ? -x : x; }
static int imax(int a, int b) { return a > b ? a : b; }
static int isup(char c) { return c >= 'A' && c <= 'Z'; }
static int islo(char c) { return c >= 'a' && c <= 'z'; }

/* ------------------------------------------------------------------ */
/* Position                                                            */
/* ------------------------------------------------------------------ */
typedef struct { int i, j; char prom; } Move;   /* prom: 0 or NBRQ */
typedef struct {
    char b[120];
    int score;
    unsigned char wc0, wc1, bc0, bc1;
    int ep, kp;
    uint64_t h;   /* content hash; pure function of the fields above,
                     sealed at construction (pos_seal).  NOT part of
                     equality -- only a constant-time reject in front of
                     the exact compare, so lookups stay provably identical
                     to Python's (hash probe, then full == on the key). */
} Pos;

static uint64_t mix64(uint64_t x) {            /* splitmix64 finalizer */
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 32; return x;
}
static void pos_seal(Pos *p) {
    /* One hash per Pos construction (rotate/fen/reset), instead of one
     * FNV over 120 bytes per table operation.  Board = exactly 15 words. */
    uint64_t w[15], h = 0;
    memcpy(w, p->b, 120);
    for (int k = 0; k < 15; k++) h = h * 0x100000001b3ULL ^ mix64(w[k] + k);
    h = h * 0x100000001b3ULL ^ mix64(((uint64_t)(uint32_t)p->score << 8)
        | (uint64_t)(p->wc0 | p->wc1 << 1 | p->bc0 << 2 | p->bc1 << 3));
    h = h * 0x100000001b3ULL ^ mix64(((uint64_t)(uint32_t)p->ep << 32)
        | (uint32_t)p->kp);
    p->h = h;
}

static int pos_eq(const Pos *a, const Pos *b) {
    /* namedtuple equality: ALL fields, score included (a position rebuilt
     * under a different K-table is a different dict key in Python too).
     * a->h == b->h is a derived-value fast reject, never a substitute. */
    return a->h == b->h && memcmp(a->b, b->b, 120) == 0 && a->score == b->score
        && a->wc0 == b->wc0 && a->wc1 == b->wc1
        && a->bc0 == b->bc0 && a->bc1 == b->bc1
        && a->ep == b->ep && a->kp == b->kp;
}

static Pos rotate(const Pos *p, int nullmove) {
    Pos r;
    for (int k = 0; k < 120; k++) {
        char c = p->b[119 - k];
        r.b[k] = isup(c) ? c + 32 : islo(c) ? c - 32 : c;
    }
    r.score = -p->score;
    r.wc0 = p->bc0; r.wc1 = p->bc1;
    r.bc0 = p->wc0; r.bc1 = p->wc1;
    r.ep = (p->ep && !nullmove) ? 119 - p->ep : 0;
    r.kp = (p->kp && !nullmove) ? 119 - p->kp : 0;
    pos_seal(&r);
    return r;
}

static int value(const Pos *p, Move m) {
    int i = m.i, j = m.j;
    char P = p->b[i], q = p->b[j];
    int sc = PSTP[(int)P][j] - PSTP[(int)P][i];
    if (islo(q)) sc += PSTP[q - 32][119 - j];
    if (iabs(j - p->kp) < 2) sc += PSTP['K'][119 - j];
    if (P == 'K' && iabs(i - j) == 2) {
        sc += PSTP['R'][(i + j) / 2];           /* i,j > 0: // == / */
        sc -= PSTP['R'][j < i ? A1 : H1];
    }
    if (P == 'P') {
        if (A8 <= j && j <= H8) sc += PSTP[(int)m.prom][j] - PSTP['P'][j];
        if (j == p->ep) sc += PSTP['P'][119 - (j + Sd)];
    }
    return sc;
}

static Pos domove(const Pos *p, Move m) {
    int i = m.i, j = m.j;
    char P = p->b[i];
    Pos n = *p;
    n.score = p->score + value(p, m);
    n.b[j] = n.b[i];
    n.b[i] = '.';
    n.wc0 = p->wc0 && i != A1; n.wc1 = p->wc1 && i != H1;
    n.bc0 = p->bc0 && j != H8; n.bc1 = p->bc1 && j != A8;
    n.ep = 0; n.kp = 0;
    if (P == 'K') {
        n.wc0 = n.wc1 = 0;
        if (iabs(j - i) == 2) {
            n.kp = (i + j) / 2;
            n.b[j < i ? A1 : H1] = '.';
            n.b[n.kp] = 'R';
        }
    }
    if (P == 'P') {
        if (A8 <= j && j <= H8) n.b[j] = m.prom;
        if (j - i == 2 * Nd) n.ep = i + Nd;
        if (j == p->ep) n.b[j + Sd] = '.';
    }
    return rotate(&n, 0);
}

/* gen_moves as a callback walk, preserving Python's exact yield order.
 * cb returning nonzero stops the walk (Python generator early exit).
 *
 * The yield ORDER is contractual (board scan 0..119, direction-list
 * order, ray order, NBRQ promotions); the TESTS are not, so they run on
 * precomputed tables: CLS classifies a square char in one load, PC_DIRS/
 * PC_ND replace the per-piece switch, and pawns get a specialized block
 * that emits the identical sequence for DIR_P = {N, N+N, N+W, N+E}. */
enum { CL_STOP = 1, CL_LOWER = 2, CL_SINGLE = 4 };  /* ' ','\n',upper | a-z | P,N,K */
static unsigned char CLS[128];
static const int *PC_DIRS[128];
static unsigned char PC_ND[128];
static void gen_init(void) {
    CLS[' '] = CLS['\n'] = CL_STOP;
    for (int c = 'A'; c <= 'Z'; c++) CLS[c] = CL_STOP;
    for (int c = 'a'; c <= 'z'; c++) CLS[c] = CL_LOWER;
    CLS['P'] |= CL_SINGLE; CLS['N'] |= CL_SINGLE; CLS['K'] |= CL_SINGLE;
    PC_DIRS['P'] = DIR_P; PC_ND['P'] = 4;
    PC_DIRS['N'] = DIR_N; PC_ND['N'] = 8;
    PC_DIRS['B'] = DIR_B; PC_ND['B'] = 4;
    PC_DIRS['R'] = DIR_R; PC_ND['R'] = 4;
    PC_DIRS['Q'] = DIR_Q; PC_ND['Q'] = 8;
    PC_DIRS['K'] = DIR_Q; PC_ND['K'] = 8;
}
typedef int (*movecb)(Move, void *);
#define YIELD(mi, mj, mp) do {                                          \
        Move _m = { (mi), (mj), (mp) };                                 \
        if (cb(_m, ctx)) return 1;                                      \
    } while (0)
#define YIELD_PAWN(mi, mj) do {                                         \
        if (A8 <= (mj) && (mj) <= H8) {                                 \
            YIELD(mi, mj, 'N'); YIELD(mi, mj, 'B');                     \
            YIELD(mi, mj, 'R'); YIELD(mi, mj, 'Q');                     \
        } else YIELD(mi, mj, 0);                                        \
    } while (0)
static int gen_moves(const Pos *p, movecb cb, void *ctx) {
    for (int i = 0; i < 120; i++) {
        char P = p->b[i];
        if (!isup(P)) continue;
        if (P == 'P') {
            /* d = N: push, only onto '.' (lower/upper/pad all break). */
            int j = i + Nd;
            if (p->b[j] == '.') YIELD_PAWN(i, j);
            /* d = N+N: from the home rank through an empty square onto
             * '.'; a double push can never reach the last rank. */
            j = i + Nd + Nd;
            if (p->b[j] == '.' && i >= A1 + Nd && p->b[i + Nd] == '.')
                YIELD(i, j, 0);
            /* d = N+W then N+E: capture, or en passant / king-passant. */
            j = i + Nd + Wd;
            if ((CLS[(int)p->b[j]] & CL_LOWER)
                || (p->b[j] == '.' && (j == p->ep || iabs(j - p->kp) <= 1)))
                YIELD_PAWN(i, j);
            j = i + Nd + Ed;
            if ((CLS[(int)p->b[j]] & CL_LOWER)
                || (p->b[j] == '.' && (j == p->ep || iabs(j - p->kp) <= 1)))
                YIELD_PAWN(i, j);
            continue;
        }
        const int *ds = PC_DIRS[(int)P];
        int nd = PC_ND[(int)P];
        int single = CLS[(int)P] & CL_SINGLE;
        for (int di = 0; di < nd; di++) {
            int d = ds[di];
            for (int j = i + d;; j += d) {
                unsigned char cq = CLS[(int)p->b[j]];
                if (cq & CL_STOP) break;
                YIELD(i, j, 0);
                if (single || (cq & CL_LOWER)) break;
                if (i == A1 && p->b[j + Ed] == 'K' && p->wc0)
                    YIELD(j + Ed, j + Wd, 0);
                if (i == H1 && p->b[j + Wd] == 'K' && p->wc1)
                    YIELD(j + Wd, j + Ed, 0);
            }
        }
    }
    return 0;
}

struct kcctx { const Pos *p; Move m; int found; };
static int kc_cb(Move m, void *vc) {
    struct kcctx *c = vc;
    if (c->p->b[m.j] == 'k' || iabs(m.j - c->p->kp) < 2) {
        c->m = m; c->found = 1; return 1;
    }
    return 0;
}
static int king_capture(const Pos *p, Move *out) {
    struct kcctx c = { p, { 0, 0, 0 }, 0 };
    gen_moves(p, kc_cb, &c);
    if (c.found && out) *out = c.m;
    return c.found;
}

/* ------------------------------------------------------------------ */
/* Insertion-ordered hash map (Python dict semantics)                  */
/* ------------------------------------------------------------------ */
/* Node storage is split hot/cold: chain walks in map_find_h touch only
 * the 16-byte hot array (hash, chain link, depth); the cold array (the
 * 140-byte Pos key, payloads, insertion-order links) is read once, on a
 * confirmed hash match.  Same dict semantics, fewer cache misses. */
typedef struct { uint64_t h; int nxt; int depth; } MHot;
typedef struct {
    Pos pos;                     /* key (depth lives in MHot) */
    int lower, upper;            /* tp_score payload */
    Move mv;                     /* tp_move payload */
    int iprev, inext;            /* insertion-order list */
    char used;
} MCold;

typedef struct {
    MHot *hot; MCold *cold; int cap;
    int *bk; long nbk;
    long count;
    int ihead, itail;
    int freehead;
} Map;

static uint64_t hash_key(const Pos *p, int depth) {
    /* p->h is sealed at construction; fold the depth in cheaply.  The
     * bucket layout this induces is unobservable: keys are unique, chain
     * hits are confirmed by pos_eq, and iteration order lives in the
     * separate insertion-order list. */
    return p->h ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(depth + 1));
}

static void map_init(Map *m) {
    m->cap = 0; m->hot = NULL; m->cold = NULL;
    m->nbk = 1 << 12;
    m->bk = malloc(sizeof(int) * m->nbk);
    for (long k = 0; k < m->nbk; k++) m->bk[k] = -1;
    m->count = 0; m->ihead = m->itail = -1; m->freehead = -1;
}
static void map_clear(Map *m) {
    free(m->hot); free(m->cold); free(m->bk);
    map_init(m);
}
static int map_find_h(Map *m, const Pos *p, int depth, uint64_t h) {
    for (int idx = m->bk[h & (m->nbk - 1)]; idx >= 0; idx = m->hot[idx].nxt)
        if (m->hot[idx].h == h && m->hot[idx].depth == depth
                && pos_eq(&m->cold[idx].pos, p))
            return idx;
    return -1;
}
static int map_find(Map *m, const Pos *p, int depth) {
    return map_find_h(m, p, depth, hash_key(p, depth));
}
static void map_rehash(Map *m) {
    long nn = m->nbk * 2;
    int *nb = malloc(sizeof(int) * nn);
    for (long k = 0; k < nn; k++) nb[k] = -1;
    for (int idx = m->ihead; idx >= 0; idx = m->cold[idx].inext) {
        long b = m->hot[idx].h & (nn - 1);
        m->hot[idx].nxt = nb[b];
        nb[b] = idx;
    }
    free(m->bk); m->bk = nb; m->nbk = nn;
}
/* Insert or update.  An update keeps its insertion slot (Python dicts). */
static int map_put(Map *m, const Pos *p, int depth) {
    uint64_t h = hash_key(p, depth);
    int idx = map_find_h(m, p, depth, h);
    if (idx >= 0) return idx;
    if (m->count + 1 > m->nbk * 3 / 4) map_rehash(m);
    if (m->freehead >= 0) {
        idx = m->freehead;
        m->freehead = m->hot[idx].nxt;
    } else {
        if (m->count >= m->cap) {
            int ncap = m->cap ? m->cap * 2 : 1 << 12;
            m->hot = realloc(m->hot, sizeof(MHot) * ncap);
            m->cold = realloc(m->cold, sizeof(MCold) * ncap);
            m->cap = ncap;
        }
        idx = (int)m->count;
        /* count == number in use; with a freelist the next fresh slot is
         * the high-water mark, tracked separately below. */
    }
    MHot *hn = &m->hot[idx];
    MCold *n = &m->cold[idx];
    n->pos = *p; hn->depth = depth;
    hn->h = h;
    long b = h & (m->nbk - 1);
    hn->nxt = m->bk[b]; m->bk[b] = idx;
    n->iprev = m->itail; n->inext = -1;
    if (m->itail >= 0) m->cold[m->itail].inext = idx; else m->ihead = idx;
    m->itail = idx;
    n->used = 1;
    m->count++;
    return idx;
}
static void map_del(Map *m, int idx) {
    MHot *hn = &m->hot[idx];
    MCold *n = &m->cold[idx];
    long b = hn->h & (m->nbk - 1);
    if (m->bk[b] == idx) m->bk[b] = hn->nxt;
    else {
        for (int k = m->bk[b]; k >= 0; k = m->hot[k].nxt)
            if (m->hot[k].nxt == idx) { m->hot[k].nxt = hn->nxt; break; }
    }
    if (n->iprev >= 0) m->cold[n->iprev].inext = n->inext; else m->ihead = n->inext;
    if (n->inext >= 0) m->cold[n->inext].iprev = n->iprev; else m->itail = n->iprev;
    n->used = 0;
    hn->nxt = m->freehead; m->freehead = idx;
    m->count--;
}

/* ------------------------------------------------------------------ */
/* Searcher state                                                      */
/* ------------------------------------------------------------------ */
static Map tps;                  /* tp_score: (pos, depth) -> (lower, upper) */
static Map tpm;                  /* tp_move:  pos -> Move                    */
static long nodes;
static long node_cap;            /* 0 = off; checked every 2048 nodes */
static double deadline;          /* seconds, monotonic; 0 = off */
static jmp_buf stopjmp;

#define MAXHIST 4096
static Pos hist[MAXHIST];
static int nhist;
static char side0;               /* side to move of hist[0]: 'w' or 'b' */
static Pos rootpos;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int in_history(const Pos *p) {
    for (int k = 0; k < nhist; k++)
        if (pos_eq(&hist[k], p)) return 1;
    return 0;
}

static int tpm_get(const Pos *p, Move *out) {
    int idx = map_find(&tpm, p, 0);
    if (idx < 0) return 0;
    *out = tpm.cold[idx].mv;
    return 1;
}
static void tpm_store(const Pos *p, Move m) {
    int idx = map_put(&tpm, p, 0);
    tpm.cold[idx].mv = m;
    if (tpm.count > TABLE_SIZE) {
        /* del next(k for k in tp_move if k != root): oldest non-root key */
        for (int k = tpm.ihead; k >= 0; k = tpm.cold[k].inext)
            if (!pos_eq(&tpm.cold[k].pos, &rootpos)) { map_del(&tpm, k); break; }
    }
}

/* Sorted move list for the main loop.  Python sorts (val, Move) tuples
 * descending; keys are unique so any sort algorithm yields one order.
 * The tuple is packed into one uint64 -- biased val in the high half,
 * then i, j, prom -- so unsigned uint64 order IS the tuple order
 * (fields are disjoint and compared most-significant first, and prom
 * compares as a byte: '\0' < 'B' < 'N' < 'Q' < 'R').  A descending
 * insertion sort on 8-byte keys beats qsort's indirect calls at these
 * sizes (typically < 64 moves). */
#define MAXMOVES 512                 /* > any pseudo-legal move count */
#define PACK_VM(val, m) (((uint64_t)((uint32_t)(val) ^ 0x80000000u) << 32) \
        | ((uint64_t)(m).i << 16) | ((uint64_t)(m).j << 8)                 \
        | (uint8_t)(m).prom)
#define VM_VAL(k)  ((int)((uint32_t)((k) >> 32) ^ 0x80000000u))
#define VM_MOVE(k) ((Move){ (int)((k) >> 16 & 0xff), (int)((k) >> 8 & 0xff), \
                            (char)((k) & 0xff) })
struct collectctx { const Pos *p; int val_lower; uint64_t *v; int n; };
static int collect_cb(Move m, void *vc) {
    struct collectctx *c = vc;
    int val = value(c->p, m);
    if (val >= c->val_lower) {
        if (c->n >= MAXMOVES) {                 /* never hide errors */
            fprintf(stderr, "ctwin: move list overflow\n");
            abort();
        }
        c->v[c->n++] = PACK_VM(val, m);
    }
    return 0;
}
static void vm_sort(uint64_t *v, int n) {       /* descending */
    for (int i = 1; i < n; i++) {
        uint64_t k = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] < k) { v[j + 1] = v[j]; j--; }
        v[j + 1] = k;
    }
}

struct termctx { const Pos *p; int all; };
static int term_cb(Move m, void *vc) {
    struct termctx *c = vc;
    Pos child = domove(c->p, m);
    if (!king_capture(&child, NULL)) { c->all = 0; return 1; }
    return 0;
}

static int has_big_piece(const Pos *p) {       /* any(c in board for "RBNQ") */
    return memchr(p->b, 'R', 120) || memchr(p->b, 'B', 120)
        || memchr(p->b, 'N', 120) || memchr(p->b, 'Q', 120);
}

/* ------------------------------------------------------------------ */
/* bound(): transcription of Searcher.bound in sunfish.py.             */
/* The generator phases run inline; PROCESS is the consumer loop body. */
/* ------------------------------------------------------------------ */
static int bound(const Pos *pos, int gamma, int depth, int root) {
    nodes++;
    if (nodes % 2048 == 0) {
        if (node_cap && nodes > node_cap) longjmp(stopjmp, 1);
        if (deadline != 0.0 && now_s() > deadline) longjmp(stopjmp, 1);
    }

    if (depth < 0) depth = 0;

    if (pos->score <= -MATE_LOWER)
        return -MATE_UPPER;

    int elow = -MATE_UPPER, eupp = MATE_UPPER;
    if (!root) {
        int idx = map_find(&tps, pos, depth);
        if (idx >= 0) { elow = tps.cold[idx].lower; eupp = tps.cold[idx].upper; }
        if (elow >= gamma) return elow;
        if (eupp < gamma) return eupp;
        if (depth > 0 && in_history(pos)) return 0;
    }

    int best = -MATE_UPPER, live = 0, done = 0;
    Move nomove = { 0, 0, 0 };

#define PROCESS(hasmv, mv, sc) do {                                     \
        int _s = (sc);                                                  \
        if (_s > best) best = _s;                                       \
        if ((hasmv) && _s > -MATE_UPPER) live = 1;                      \
        if (best >= gamma) {                                            \
            if ((hasmv) && depth) tpm_store(pos, (mv));                 \
            done = 1;                                                   \
        }                                                               \
    } while (0)

    /* moves() first statement: read the killer BEFORE the null move. */
    Move killer = nomove;
    int have_killer = tpm_get(pos, &killer);

    /* Null move, capped at static eval plus one score bucket. */
    if (!root && depth > NULL_MIN_DEPTH && iabs(pos->score) < NULL_LIMIT
            && has_big_piece(pos)) {
        Pos rp = rotate(pos, 1);
        int s = -bound(&rp, 1 - gamma, depth - NULL_RED, 0);
        int margin = NULL_MARGIN < 0 ? EVAL_ROUGHNESS : NULL_MARGIN;
        int score = pos->score + margin;
        if (s < score) score = s;
        Move proof = nomove;
        int have_proof = 0;
        if (score >= gamma) {                       /* short-circuit `and` */
            have_proof = tpm_get(pos, &proof);       /* re-read, like Python */
            if (!have_proof) have_proof = king_capture(pos, &proof);
        }
        if (have_proof && value(pos, proof) >= MATE_LOWER)
            PROCESS(1, proof, MATE_UPPER);
        else
            PROCESS(0, nomove, score);
        if (done) goto after_moves;
    }

    /* QSearch stand pat. */
    if (depth == 0) {
        PROCESS(0, nomove, pos->score);
        if (done) goto after_moves;
    }

    /* Internal iterative deepening (driver probe: root=1, unstored). */
    if (!have_killer && depth > IID_MIN_DEPTH) {
        bound(pos, gamma, depth - IID_RED, 1);
        have_killer = tpm_get(pos, &killer);
    }

    int val_lower = QS - depth * QS_A;

    /* Killer first, gated by the QS threshold. */
    if (have_killer && value(pos, killer) >= val_lower) {
        Pos np = domove(pos, killer);
        PROCESS(1, killer, -bound(&np, 1 - gamma, depth - 1, 0));
        if (done) goto after_moves;
    }

    /* Then all moves above the threshold, sorted by descending value. */
    {
        uint64_t vbuf[MAXMOVES];             /* stack: longjmp-safe, no malloc */
        struct collectctx c = { pos, val_lower, vbuf, 0 };
        gen_moves(pos, collect_cb, &c);
        vm_sort(vbuf, c.n);
        for (int k = 0; k < c.n; k++) {
            int val = VM_VAL(vbuf[k]);
            Move m = VM_MOVE(vbuf[k]);
            if (depth <= FUT_MAX && pos->score + val < gamma) {
                /* Futility: value evidence only, except the mate special
                 * case, which is a real (cutting) witness. */
                if (val >= MATE_LOWER) PROCESS(1, m, MATE_UPPER);
                else PROCESS(0, nomove, pos->score + val);
                break;                       /* Python breaks either way */
            }
            Pos np = domove(pos, m);
            PROCESS(1, m, -bound(&np, 1 - gamma, depth - 1, 0));
            if (done) break;
        }
    }

after_moves:
    /* Only virtual evidence seen: classify mate/stalemate exactly. */
    if (depth && !live) {
        struct termctx tc = { pos, 1 };
        gen_moves(pos, term_cb, &tc);
        if (tc.all) {
            Pos rp = rotate(pos, 1);
            if (king_capture(&rp, NULL))
                best = MATE_DIST
                     ? imax(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)
                     : -MATE_LOWER;
            else
                best = 0;
        }
    }

    if (!root) {
        int idx = map_put(&tps, pos, depth);
        if (best >= gamma) { tps.cold[idx].lower = best; tps.cold[idx].upper = eupp; }
        else               { tps.cold[idx].lower = elow; tps.cold[idx].upper = best; }
    }
    if (tps.count > TABLE_SIZE)
        map_del(&tps, tps.ihead);

    return best;
#undef PROCESS
}

/* ------------------------------------------------------------------ */
/* MTD-bi driver: transcription of Searcher.search                     */
/* ------------------------------------------------------------------ */
static void fmt_move(char *buf, const Move *m, int have) {
    if (!have) { strcpy(buf, "-"); return; }
    if (m->prom) sprintf(buf, "%d,%d,%c", m->i, m->j, m->prom);
    else sprintf(buf, "%d,%d,-", m->i, m->j);
}

static void search_setup(void) {
    nodes = 0;
    map_clear(&tps);
    rootpos = hist[nhist - 1];
    PSTP['K'] = (memchr(rootpos.b, 'Q', 120) && memchr(rootpos.b, 'q', 120))
              ? TAB[5] : KEND;
}

/* Fixed-depth probe loop for the differential harness: identical yield
 * sequence to Searcher.search, stopped when depth==maxd converges. */
static void go_depth(int maxd) {
    search_setup();
    node_cap = 0; deadline = 0.0;
    if (setjmp(stopjmp)) return;             /* unreachable with caps off */
    int gamma = 0;
    long last_nodes = 0;
    for (int depth = 1; depth <= maxd && depth < 1000; depth++) {
        int lower = 1 - MATE_UPPER, upper = MATE_UPPER;
        while (lower < upper - EVAL_ROUGHNESS) {
            int score = bound(&rootpos, gamma, depth, 1);
            if (score >= gamma) lower = score;
            if (score < gamma) upper = score;
            Move mv; char mb[32];
            fmt_move(mb, &mv, tpm_get(&rootpos, &mv));
            printf("info depth %d gamma %d score %d move %s nodes %ld\n",
                   depth, gamma, score, mb, nodes);
            last_nodes = nodes;
            gamma = pyfloordiv(lower + upper + 1, 2);
        }
    }
    printf("done nodes %ld\n", last_nodes);
    fflush(stdout);
}

/* Game-loop go for surrogate matches: fixed nodes (primary), or movetime.
 * Structure mirrors the classic main() driver: candidates only from
 * fail-highs, committed when their depth completes. */
static void render_sq(char *buf, int i) {
    buf[0] = (char)('a' + pymod(i - A1, 10));
    buf[1] = (char)('0' + (1 - pyfloordiv(i - A1, 10)));
}
static void go_game(long max_nodes, double movetime_s, int maxd) {
    search_setup();
    node_cap = max_nodes;
    double start = now_s();
    deadline = movetime_s > 0 ? start + (movetime_s > 0.05 ? movetime_s : 0.05) : 0.0;
    char best[8] = "", cand[8] = "";
    int d0 = 1;
    int mover_black = (side0 == 'b') ^ ((nhist - 1) % 2);

    if (!setjmp(stopjmp)) {
        int gamma = 0;
        for (int depth = 1; depth < 1000 && depth <= maxd; depth++) {
            int lower = 1 - MATE_UPPER, upper = MATE_UPPER;
            while (lower < upper - EVAL_ROUGHNESS) {
                int score = bound(&rootpos, gamma, depth, 1);
                if (score >= gamma) lower = score;
                if (score < gamma) upper = score;
                /* --- yield consumer (classic main go loop) --- */
                if (depth > d0) {
                    if (cand[0]) strcpy(best, cand);
                    d0 = depth;
                }
                if (max_nodes && nodes >= max_nodes && (best[0] || cand[0]))
                    goto out;
                if (score >= gamma) {
                    Move mv;
                    if (!tpm_get(&rootpos, &mv)) {
                        printf("info depth %d score cp %d\n", depth, score);
                        goto out;
                    }
                    int i = mv.i, j = mv.j;
                    if (mover_black) { i = 119 - i; j = 119 - j; }
                    render_sq(cand, i); render_sq(cand + 2, j);
                    cand[4] = mv.prom ? mv.prom + 32 : 0;
                    cand[5] = 0;
                    printf("info depth %d score cp %d pv %s\n", depth, score, cand);
                }
                if ((best[0] || cand[0]) && deadline != 0.0
                        && now_s() - start > (deadline - start) * 0.8)
                    goto out;
                gamma = pyfloordiv(lower + upper + 1, 2);
            }
        }
    }
out:
    node_cap = 0; deadline = 0.0;
    printf("bestmove %s\n", best[0] ? best : cand[0] ? cand : "(none)");
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Position setup                                                      */
/* ------------------------------------------------------------------ */
static void reset_state(void) {
    map_clear(&tps);
    map_clear(&tpm);
    nodes = 0;
    PSTP['K'] = TAB[5];
    memcpy(hist[0].b, INITIAL, 120);
    hist[0].score = 0;
    hist[0].wc0 = hist[0].wc1 = hist[0].bc0 = hist[0].bc1 = 1;
    hist[0].ep = hist[0].kp = 0;
    pos_seal(&hist[0]);
    nhist = 1;
    side0 = 'w';
}

static int parse_sq(const char *c) {
    return A1 + (c[0] - 'a') - 10 * (c[1] - '1');
}

static int setup_fen(char **tok, int ntok) {
    /* tok: placement side castling ep [halfmove fullmove] -- mirrors
     * pyref.from_fen exactly. */
    if (ntok < 4) return 0;
    char board[120];
    int k = 0;
    for (int r = 0; r < 20; r++) board[k++] = (r % 10 == 9) ? '\n' : ' ';
    const char *pl = tok[0];
    for (int row = 0; row < 8; row++) {
        board[k++] = ' ';
        int files = 0;
        for (; *pl && *pl != '/'; pl++) {
            if (*pl >= '1' && *pl <= '8')
                for (int e = 0; e < *pl - '0'; e++) { board[k++] = '.'; files++; }
            else { board[k++] = *pl; files++; }
        }
        if (*pl == '/') pl++;
        if (files != 8) return 0;
        board[k++] = '\n';
    }
    for (int r = 0; r < 20; r++) board[k++] = (r % 10 == 9) ? '\n' : ' ';
    if (k != 120) return 0;

    Pos p;
    memcpy(p.b, board, 120);
    p.wc0 = strchr(tok[2], 'Q') != NULL; p.wc1 = strchr(tok[2], 'K') != NULL;
    p.bc0 = strchr(tok[2], 'k') != NULL; p.bc1 = strchr(tok[2], 'q') != NULL;
    p.ep = strcmp(tok[3], "-") ? parse_sq(tok[3]) : 0;
    p.kp = 0;
    int score = 0;
    for (int i = 0; i < 120; i++) {
        char c = p.b[i];
        if (isup(c)) score += PSTP[(int)c][i];
        else if (islo(c)) score -= PSTP[c - 32][119 - i];
    }
    p.score = score;
    pos_seal(&p);
    side0 = tok[1][0];
    if (side0 == 'b') p = rotate(&p, 0);
    hist[0] = p;
    nhist = 1;
    return 1;
}

static void apply_uci_moves(char **tok, int ntok) {
    for (int ply = 0; ply < ntok; ply++) {
        const char *mv = tok[ply];
        Move m;
        m.i = parse_sq(mv);
        m.j = parse_sq(mv + 2);
        m.prom = mv[4] ? mv[4] - 32 : 0;      /* 'q' -> 'Q' */
        int black = (side0 == 'b') ? (ply % 2 == 0) : (ply % 2 == 1);
        if (black) { m.i = 119 - m.i; m.j = 119 - m.j; }
        hist[nhist] = domove(&hist[nhist - 1], m);
        nhist++;
    }
}

/* ------------------------------------------------------------------ */
/* Table loading and knobs                                             */
/* ------------------------------------------------------------------ */
static int load_tables(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[8192];
    int seen = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char *tok = strtok(line, " \t\n");
        if (!tok) continue;
        if (!strcmp(tok, "piece")) {
            for (int k = 0; k < 6; k++) PIECEVAL[k] = atoi(strtok(NULL, " \t\n"));
            seen |= 1;
        } else if (!strcmp(tok, "KEND")) {
            for (int k = 0; k < 120; k++) KEND[k] = atoi(strtok(NULL, " \t\n"));
            seen |= 2;
        } else {
            const char *order = "PNBRQK";
            const char *at = strchr(order, tok[0]);
            if (at && tok[1] == 0) {
                int pi = (int)(at - order);
                for (int k = 0; k < 120; k++)
                    TAB[pi][k] = atoi(strtok(NULL, " \t\n"));
                seen |= 4 << pi;
            }
        }
    }
    fclose(f);
    if (seen != (1 | 2 | 4 | 8 | 16 | 32 | 64 | 128)) return 0;
    for (int c = 0; c < 128; c++) PSTP[c] = NULL;
    const char *order = "PNBRQK";
    for (int k = 0; k < 6; k++) PSTP[(int)order[k]] = TAB[k];
    MATE_LOWER = PIECEVAL[5] - 13 * PIECEVAL[4];
    MATE_UPPER = PIECEVAL[5] + 10 * PIECEVAL[4];
    tables_loaded = 1;
    return 1;
}

struct knob { const char *name; int *ip; long *lp; };
static struct knob KNOBS[] = {
    { "QS", &QS, NULL }, { "QS_A", &QS_A, NULL },
    { "EVAL_ROUGHNESS", &EVAL_ROUGHNESS, NULL },
    { "TABLE_SIZE", NULL, &TABLE_SIZE },
    { "NULL_MARGIN", &NULL_MARGIN, NULL },
    { "NULL_MIN_DEPTH", &NULL_MIN_DEPTH, NULL },
    { "NULL_LIMIT", &NULL_LIMIT, NULL },
    { "NULL_RED", &NULL_RED, NULL },
    { "IID_MIN_DEPTH", &IID_MIN_DEPTH, NULL },
    { "IID_RED", &IID_RED, NULL },
    { "FUT_MAX", &FUT_MAX, NULL },
    { "MATE_DIST", &MATE_DIST, NULL },
    { NULL, NULL, NULL }
};
static int set_knob(const char *name, long v) {
    for (struct knob *k = KNOBS; k->name; k++)
        if (!strcmp(k->name, name)) {
            if (k->ip) *k->ip = (int)v; else *k->lp = v;
            return 1;
        }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Protocol loop                                                       */
/* ------------------------------------------------------------------ */
struct listctx { const Pos *p; };
static int list_cb(Move m, void *vc) {
    struct listctx *c = vc;
    char mb[32];
    fmt_move(mb, &m, 1);
    printf("mv %s %d\n", mb, value(c->p, m));
    return 0;
}

int main(int argc, char **argv) {
    gen_init();
    const char *tpath = argc > 1 ? argv[1] : getenv("SF_TABLES");
    if (tpath && !load_tables(tpath)) {
        fprintf(stderr, "ctwin: cannot load tables from %s\n", tpath);
        return 1;
    }
    /* env knobs: SF_QS=.. etc. */
    for (struct knob *k = KNOBS; k->name; k++) {
        char env[64];
        snprintf(env, sizeof env, "SF_%s", k->name);
        const char *v = getenv(env);
        if (v) set_knob(k->name, atol(v));
    }
    if (tables_loaded) reset_state();

    char line[65536];
    while (fgets(line, sizeof line, stdin)) {
        char *tok[8192];
        int ntok = 0;
        for (char *t = strtok(line, " \t\n"); t && ntok < 8192; t = strtok(NULL, " \t\n"))
            tok[ntok++] = t;
        if (!ntok) continue;

        if (!strcmp(tok[0], "quit")) break;

        else if (!strcmp(tok[0], "tables")) {
            if (ntok > 1 && load_tables(tok[1])) { reset_state(); puts("ok"); }
            else puts("err tables");
        }

        else if (!tables_loaded) {
            puts("err no tables loaded");
        }

        else if (!strcmp(tok[0], "reset") || !strcmp(tok[0], "ucinewgame")) {
            reset_state();
            if (strcmp(tok[0], "ucinewgame")) puts("ok");
        }

        else if (!strcmp(tok[0], "set")) {
            if (ntok >= 3 && set_knob(tok[1], atol(tok[2]))) puts("ok");
            else puts("err knob");
        }

        else if (!strcmp(tok[0], "position")) {
            if (ntok >= 2 && !strcmp(tok[1], "startpos")) {
                reset_ok:
                memcpy(hist[0].b, INITIAL, 120);
                hist[0].score = 0;
                hist[0].wc0 = hist[0].wc1 = hist[0].bc0 = hist[0].bc1 = 1;
                hist[0].ep = hist[0].kp = 0;
                pos_seal(&hist[0]);
                nhist = 1;
                side0 = 'w';
                if (ntok >= 3 && !strcmp(tok[2], "moves"))
                    apply_uci_moves(tok + 3, ntok - 3);
                puts("ok");
            } else if (ntok >= 2 && !strcmp(tok[1], "fen")) {
                int mstart = ntok;
                for (int k = 2; k < ntok; k++)
                    if (!strcmp(tok[k], "moves")) { mstart = k; break; }
                if (setup_fen(tok + 2, mstart - 2)) {
                    if (mstart < ntok) apply_uci_moves(tok + mstart + 1, ntok - mstart - 1);
                    puts("ok");
                } else puts("err fen");
            } else goto reset_ok;
        }

        else if (!strcmp(tok[0], "push")) {
            Move m;
            m.i = atoi(tok[1]); m.j = atoi(tok[2]);
            m.prom = strcmp(tok[3], "-") ? tok[3][0] : 0;
            hist[nhist] = domove(&hist[nhist - 1], m);
            nhist++;
            puts("ok");
        }

        else if (!strcmp(tok[0], "pop")) {
            if (nhist > 1) nhist--;
            puts("ok");
        }

        else if (!strcmp(tok[0], "moves")) {
            struct listctx c = { &hist[nhist - 1] };
            gen_moves(&hist[nhist - 1], list_cb, &c);
            puts("end");
        }

        else if (!strcmp(tok[0], "go")) {
            long gnodes = 0; double mt = 0; int gdepth = 0;
            for (int k = 1; k + 1 < ntok; k += 2) {
                if (!strcmp(tok[k], "depth")) gdepth = atoi(tok[k + 1]);
                else if (!strcmp(tok[k], "nodes")) gnodes = atol(tok[k + 1]);
                else if (!strcmp(tok[k], "movetime")) mt = atol(tok[k + 1]) / 1000.0;
            }
            if (gdepth && !gnodes && mt == 0) go_depth(gdepth);
            else go_game(gnodes, mt, gdepth ? gdepth : 999);
        }

        else if (!strcmp(tok[0], "uci")) {
            puts("id name sunfish ctwin");
            puts("uciok");
        }
        else if (!strcmp(tok[0], "isready")) puts("readyok");
        else printf("err unknown command: %s\n", tok[0]);
        fflush(stdout);
    }
    return 0;
}
