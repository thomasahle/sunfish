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
 * Reference: sunfish.py at the repo root of this checkout (capped null
 * move, mate-distance scoring, IID at depth > 3).  Historical master deltas
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
static int REF_TAB[6][120], REF_KEND[120], REF_PIECEVAL[6];
static int *PSTP[128];           /* by piece char; PSTP['K'] is swapped */
static int MATE_LOWER, MATE_UPPER;
static int tables_loaded = 0;

/* Runtime knobs.  Defaults reproduce sunfish.py at the repo root. */
static int QS = 40;
static int QS_A = 140;
static int LMR = 75;
static int EVAL_ROUGHNESS = 15;
static int NULL_CAP_MARGIN = -1; /* -1 follows EVAL_ROUGHNESS, as Python */
static int VALUE_N = 280, VALUE_B = 320, VALUE_R = 479, VALUE_Q = 929;
static int PST_P = 100, PST_N = 100, PST_B = 100, PST_R = 100;
static int PST_Q = 100, PST_K = 100, PST_KE = 100;
static long TABLE_SIZE = 1000000;
static int NULL_MARGIN = -200;   /* fuel-probe target margin */
static int NULL_MIN_DEPTH = 2;   /* null move when depth > this */
static int NULL_LIMIT = 750;     /* |score| bound for both null mechanisms */
static int NULL_CUT_RED = 3;     /* shallow null-candidate reduction */
static int NULL_RED = 7;         /* deep fuel-probe reduction */
static int IID_MIN_DEPTH = 99;   /* tuned off; retained as a lab knob */
static int IID_RED = 3;          /* IID depth reduction */
static int FUT_MAX = 1;          /* futility pruning when depth <= this */
static int FUT_CAP = 1;          /* 0 off, 1 ordinary moves, 2 negative value */
static int FUT_CAP_DEPTH = 3;
static int MATE_DIST = 1;        /* mate scores carry distance (master: 0) */
/* Replacement-policy battery knobs (tp_move only; tp_score untouched).
 * EVICT_POLICY 0: master/branch root-guarded FIFO insert-then-evict (>).
 *              1: proposed unguarded evict-BEFORE-insert (>=) -- a
 *                 variant, not a no-op: boundary, order and guard differ.
 *              2: depth-stored bounded scan -- insert-then-evict (>),
 *                 scan the first min(EVICT_SCAN_K, len) FIFO entries and
 *                 evict the one with the shallowest LAST-store depth
 *                 (ties: the earliest scanned).  No root guard.
 *              3: hash-slot table, TABLE_SIZE buckets x two tiers per
 *                 bucket (deep slot: replace if new depth >= stored;
 *                 else an always-replace slot).  Exact-position compare
 *                 on read and update -- a colliding read returns nothing,
 *                 never a foreign move.
 * KILLER_COUNT k: keep the k most recent DISTINCT fail-high moves per
 * position (most recent first; k-deepest is the noted follow-up).  Reads:
 * single-move consumers (null proof, driver yield) take the most recent;
 * the killer search phase tries all k in order before the sorted list. */
static int EVICT_POLICY = 0;
static int EVICT_SCAN_K = 4;
static int KILLER_COUNT = 1;     /* 1..3 */
static int USE_VARIANT = 0;      /* no-op here: forces pyref's transcribed
                                    VariantSearcher so difftest can prove
                                    transcription==reference at defaults */
#define MAXKILL 3

/* PR-service knobs: exact ports of open-PR search diffs, one knob each,
 * identity-proven against the PR branch's own sunfish.py (pyref run in a
 * worktree of that branch; C-only knobs go through difftest --cset).
 * PR #171 fix-qsearch-frontier-evasions: before declaring mate at a
 *   not-live node, find the first legal move; if every legal move is
 *   below this node's QS threshold, retry ONLY the filtered tail in
 *   generator order (unsorted, no killer/IID/futility), as an unstored
 *   root probe (its PR base is IID>2/no-mate-distance: --cset
 *   IID_MIN_DEPTH=2 MATE_DIST=0 alongside QS_TAIL=1).
 * FUEL_NULL (MASTER DEFAULT since #192; born as the PR #182 port):
 *   classic capped null only for NULL_MIN_DEPTH < depth < FUEL_MIN_DEPTH;
 *   from FUEL_MIN_DEPTH a null probe at the fixed target pos.score +
 *   NULL_MARGIN is a FUEL ORACLE, never a score candidate: pass beats
 *   target => real moves spend FUEL_NULL extra depth units (nominal depth
 *   still keys tables and QS).  FUEL_NULL=0 = pre-#192 deep null.
 * PR #184 derive-never-inherit: search() re-derives every history score
 *   from the board under the K-table chosen for THIS search, so no score
 *   is inherited across table swaps. */
static int QS_TAIL = 0;
static int FUEL_NULL = 1;        /* Reduction amount; 0 restores pre-#192. */
static int FUEL_MIN_DEPTH = 6;
static int DERIVE_FRESH = 0;
/* FEN_HIST: how a `position fen` builds the history, which is a SEARCH INPUT.
 * sunfish_ui/uci.py -- the driver every match runs -- writes
 *     hist = [pos] if get_color(pos) == WHITE else [pos.rotate(), pos]
 * so a BLACK-to-move FEN starts with TWO plies: the root, preceded by its own
 * white-POV mirror.  Searcher.search does `self.history = set(hist)` and
 * bound() returns 0 for a non-root node found there, so the mirror is a live
 * draw-scoring entry from move 1 -- and the null move reaches it exactly
 * (rotate(nullmove) == rotate() whenever ep == kp == 0, i.e. for every book
 * position without an en-passant square).
 *   1 (default) = the driver's construction: what every match plays.
 *   0           = the one-ply construction this file used before.
 * pyref.py carries the same knob under the same name, so `difftest --set
 * FEN_HIST=N` keeps both sides of the identity gate honest either way. */
static int FEN_HIST = 1;

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
static long gen_calls;           /* movegen walks started (battery metric;
                                    the Python side counts gen_moves() calls
                                    the same way -- compared in `done` lines) */
static int gen_moves(const Pos *p, movecb cb, void *ctx) {
    gen_calls++;
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
    Move mvs[MAXKILL];           /* tp_move payload: most recent first */
    int mds[MAXKILL];            /* last-store depth per move */
    unsigned char nmv;
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
    n->nmv = 0;                  /* fresh dict entry: empty killer list */
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
static int nhist0 = 1;           /* history length right after `position`: 2 for
                                    a black-to-move FEN under FEN_HIST=1, else 1.
                                    Move parity counts from HERE, so it holds
                                    under either construction and through the
                                    difftest push/pop walk. */
static char side0;               /* side to move of hist[nhist0-1]: 'w' or 'b' */
static Pos rootpos;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Is the side to move at hist[nhist-1] black?  Under FEN_HIST=1 this is the
 * driver's own `len(hist) % 2 == 0`, stated without assuming the base length. */
static int mover_is_black(void) {
    return ((side0 == 'b') ^ ((nhist - nhist0) & 1)) != 0;
}

static int in_history(const Pos *p) {
    for (int k = 0; k < nhist; k++)
        if (pos_eq(&hist[k], p)) return 1;
    return 0;
}

static int move_eq(Move a, Move b) {
    return a.i == b.i && a.j == b.j && a.prom == b.prom;
}
/* Push m to the front of a killer list, deduplicating, keeping at most
 * KILLER_COUNT entries.  Records the store depth alongside. */
static void klist_push(Move *mvs, int *mds, unsigned char *nmv, Move m, int depth) {
    int n = *nmv, at = -1;
    for (int k = 0; k < n; k++)
        if (move_eq(mvs[k], m)) { at = k; break; }
    if (at < 0) { n = n < KILLER_COUNT ? n + 1 : KILLER_COUNT; at = n - 1; }
    for (int k = at; k > 0; k--) { mvs[k] = mvs[k - 1]; mds[k] = mds[k - 1]; }
    mvs[0] = m; mds[0] = depth;
    *nmv = (unsigned char)n;
}

/* Policy 3: fixed hash-slot table, 2 tiers per bucket.  Exact-position
 * keys: a colliding read returns nothing, never a foreign move. */
typedef struct {
    Pos pos; Move mvs[MAXKILL]; int mds[MAXKILL];
    unsigned char nmv, used;
} KSlot;
static KSlot *kslots;            /* 2 * kslot_n entries; [2b]=deep [2b+1]=always */
static long kslot_n;

static void kslot_reset(void) {
    free(kslots); kslots = NULL; kslot_n = 0;
    if (EVICT_POLICY == 3) {
        kslot_n = TABLE_SIZE > 0 ? TABLE_SIZE : 1;
        kslots = calloc(2 * kslot_n, sizeof(KSlot));
    }
}
/* Knobs are set AFTER reset in the harness flow; the slot table sizes
 * itself on first use after they settle.  A TABLE_SIZE/policy change
 * mid-session recreates it empty (documented battery semantics; the
 * Python variant recreates its table on knob set the same way). */
static void kslot_ensure(void) {
    if (!kslots || kslot_n != (TABLE_SIZE > 0 ? TABLE_SIZE : 1)) kslot_reset();
}
static KSlot *kslot_pair(const Pos *p) { return &kslots[2 * (long)(p->h % (uint64_t)kslot_n)]; }

static int tpm_get_all(const Pos *p, Move *mvs_out, int *n_out) {
    if (EVICT_POLICY == 3) {
        kslot_ensure();
        KSlot *s = kslot_pair(p);
        for (int t = 0; t < 2; t++)
            if (s[t].used && pos_eq(&s[t].pos, p)) {
                for (int k = 0; k < s[t].nmv; k++) mvs_out[k] = s[t].mvs[k];
                *n_out = s[t].nmv;
                return s[t].nmv > 0;
            }
        *n_out = 0;
        return 0;
    }
    int idx = map_find(&tpm, p, 0);
    if (idx < 0) { *n_out = 0; return 0; }
    for (int k = 0; k < tpm.cold[idx].nmv; k++) mvs_out[k] = tpm.cold[idx].mvs[k];
    *n_out = tpm.cold[idx].nmv;
    return tpm.cold[idx].nmv > 0;
}
static int tpm_get(const Pos *p, Move *out) {   /* most recent killer */
    Move mvs[MAXKILL]; int n;
    if (!tpm_get_all(p, mvs, &n)) return 0;
    *out = mvs[0];
    return 1;
}

static void tpm_store(const Pos *p, Move m, int depth) {
    if (EVICT_POLICY == 3) {
        kslot_ensure();
        KSlot *s = kslot_pair(p);
        int t;
        for (t = 0; t < 2; t++)
            if (s[t].used && pos_eq(&s[t].pos, p)) {         /* in-place update */
                klist_push(s[t].mvs, s[t].mds, &s[t].nmv, m, depth);
                return;
            }
        t = (!s[0].used || depth >= s[0].mds[0]) ? 0 : 1;    /* deep else always */
        s[t].pos = *p;
        s[t].nmv = 0;
        klist_push(s[t].mvs, s[t].mds, &s[t].nmv, m, depth);
        s[t].used = 1;
        return;
    }
    if (EVICT_POLICY == 1) {
        /* if len(tp_move) >= TABLE_SIZE: del tp_move[next(iter(tp_move))]
         * BEFORE the store -- unguarded: may evict the root, and may evict
         * the very key being stored (which then re-inserts at the tail --
         * an update that MOVES slot, unlike a plain dict update). */
        if (tpm.count >= TABLE_SIZE) map_del(&tpm, tpm.ihead);
    }
    int idx = map_put(&tpm, p, 0);
    klist_push(tpm.cold[idx].mvs, tpm.cold[idx].mds, &tpm.cold[idx].nmv, m, depth);
    if (EVICT_POLICY == 0 && tpm.count > TABLE_SIZE) {
        /* del next(k for k in tp_move if k != root): oldest non-root key */
        for (int k = tpm.ihead; k >= 0; k = tpm.cold[k].inext)
            if (!pos_eq(&tpm.cold[k].pos, &rootpos)) { map_del(&tpm, k); break; }
    }
    if (EVICT_POLICY == 2 && tpm.count > TABLE_SIZE) {
        /* bounded scan from the FIFO front: evict the shallowest
         * last-store depth among the first min(K, len); ties keep the
         * earliest scanned. */
        int victim = tpm.ihead, k = tpm.ihead, seen = 0, vd = 0;
        for (; k >= 0 && seen < EVICT_SCAN_K; k = tpm.cold[k].inext, seen++) {
            int dk = tpm.cold[k].mds[0];
            if (seen == 0 || dk < vd) { victim = k; vd = dk; }
        }
        map_del(&tpm, victim);
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
struct collectctx { const Pos *p; int val_lower; uint64_t *v; int n; int tail; };
static int collect_cb(Move m, void *vc) {
    struct collectctx *c = vc;
    int val = value(c->p, m);
    /* tail=0: admit >= threshold (classic); tail=1 (PR #171 qs_tail
     * probe): admit the complementary below-threshold tail. */
    if ((val >= c->val_lower) != c->tail) {
        if (c->n >= MAXMOVES) {                 /* never hide errors */
            fprintf(stderr, "ctwin: move list overflow\n");
            abort();
        }
        c->v[c->n++] = PACK_VM(val, m);
    }
    return 0;
}

/* PR #171 terminal scan: lazy transcription of
 *   legal = (m for m in gen_moves() if not move(m).king_capture())
 *   move = next(legal, None)
 *   ... and all(value(m) < val_lower for m in legal)
 * stop_at_first reproduces the tail-probe short-circuit (next() only). */
struct qtctx { const Pos *p; int found; int above; int val_lower; int stop_at_first; };
static int qt_cb(Move m, void *vc) {
    struct qtctx *c = vc;
    Pos child = domove(c->p, m);
    if (king_capture(&child, NULL)) return 0;       /* illegal: skip */
    if (!c->found) {
        c->found = 1;
        if (c->stop_at_first) return 1;             /* next(legal) only */
        if (value(c->p, m) >= c->val_lower) { c->above = 1; return 1; }
        return 0;
    }
    if (value(c->p, m) >= c->val_lower) { c->above = 1; return 1; }
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

static int bound(const Pos *pos, int gamma, int depth, int root, int qstail);

static int score_move(const Pos *pos, Move move, int val, int gamma,
        int depth, int rd, int root, int guard, int *real) {
    *real = 1;
    if (val >= MATE_LOWER) return MATE_UPPER;
    int capped = depth <= FUT_MAX ? val < MATE_LOWER
        : depth <= FUT_CAP_DEPTH && (FUT_CAP == 1 ? val < MATE_LOWER : FUT_CAP == 2 && val < 0);
    int cap = MATE_UPPER;
    if (capped) {
        cap = pos->score + val + (depth > 1 ? depth - 1 : 0) * QS_A;
        if (cap >= MATE_LOWER) cap = MATE_LOWER - 1;
        if (cap < gamma) { *real = 0; return cap; }
    }
    int move_depth = rd - 1 - (!root && guard && val < LMR);
    Pos child = domove(pos, move);
    int full = -bound(&child, 1 - gamma, move_depth, 0, 0);
    return cap < full ? cap : full;
}

/* ------------------------------------------------------------------ */
/* bound(): transcription of Searcher.bound in sunfish.py.             */
/* The generator phases run inline; PROCESS is the consumer loop body. */
/* ------------------------------------------------------------------ */
static int bound(const Pos *pos, int gamma, int depth, int root, int qstail) {
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

    int val_lower = depth == 0 ? QS : -MATE_UPPER;
    int best = -MATE_UPPER, live = 0, done = 0;
    Move nomove = { 0, 0, 0 };

#define PROCESS(hasmv, mv, sc) do {                                     \
        int _s = (sc);                                                  \
        if (_s > best) best = _s;                                       \
        if ((hasmv) && _s > -MATE_UPPER) live = 1;                      \
        if (best >= gamma) {                                            \
            if ((hasmv) && depth) tpm_store(pos, (mv), depth);          \
            done = 1;                                                   \
        }                                                               \
    } while (0)

    /* moves() first statement: read the killer(s) BEFORE the null move. */
    Move killers[MAXKILL];
    int nkill = 0;
    tpm_get_all(pos, killers, &nkill);

    /* Null move, capped at static eval plus one score bucket. Since #192
     * this branch runs only below FUEL_MIN_DEPTH; above it the fuel oracle
     * decides a reduction. FUEL_NULL=0 restores the old deep-null cutoff. */
    if (!root && depth > NULL_MIN_DEPTH
            && (!FUEL_NULL || depth < FUEL_MIN_DEPTH)
            && iabs(pos->score) < NULL_LIMIT && has_big_piece(pos)) {
        int score = pos->score + (NULL_CAP_MARGIN < 0
            ? EVAL_ROUGHNESS : NULL_CAP_MARGIN);
        if (score >= gamma) {
            Pos rp = rotate(pos, 1);
            int s = -bound(&rp, 1 - gamma, depth - NULL_CUT_RED, 0, 0);
            if (s < score) score = s;
        }
        Move proof = nomove;
        int have_proof = score >= gamma && king_capture(pos, &proof);
        if (have_proof)
            PROCESS(1, proof, MATE_UPPER);
        else
            PROCESS(0, nomove, score);
        if (done) goto after_moves;
    }

    /* Fuel oracle -- its fixed target reduces the node.  Its static guard
     * also limits intrinsic LMR to positions where passing is meaningful. */
    int rd = depth;
    int guard = depth >= FUEL_MIN_DEPTH
        && iabs(pos->score) < NULL_LIMIT && has_big_piece(pos);
    if (guard && FUEL_NULL) {
        int target = pos->score + NULL_MARGIN;
        Pos rp = rotate(pos, 1);
        if (-bound(&rp, 1 - target, depth - NULL_RED, 0, 0) >= target)
            rd = depth - FUEL_NULL;
    }

    /* QSearch stand pat. */
    if (depth == 0) {
        PROCESS(0, nomove, pos->score);
        if (done) goto after_moves;
    }

    /* Optional lab IID (driver probe: root=1, unstored). */
    if (!qstail && nkill == 0 && depth > IID_MIN_DEPTH) {
        bound(pos, gamma, depth - IID_RED, 1, 0);
        tpm_get_all(pos, killers, &nkill);
    }

    /* Killer(s) first, gated by the QS threshold, most recent first.
     * A qs_tail probe skips the killer phase. */
    if (!qstail)
    for (int kk = 0; kk < nkill; kk++) {
        int val = value(pos, killers[kk]);
        if (val < val_lower) continue;
        int real;
        int score = score_move(pos, killers[kk], val, gamma, depth, rd, root, guard, &real);
        PROCESS(real, killers[kk], score);
        if (done) goto after_moves;
    }

    /* Then all moves above the threshold, sorted by descending value.
     * A qs_tail probe takes the complementary tail (below-threshold
     * moves) in GENERATOR order -- unsorted, no futility. */
    {
        uint64_t vbuf[MAXMOVES];             /* stack: longjmp-safe, no malloc */
        struct collectctx c = { pos, val_lower, vbuf, 0, qstail };
        gen_moves(pos, collect_cb, &c);
        if (!qstail) vm_sort(vbuf, c.n);
        for (int k = 0; k < c.n; k++) {
            int val = VM_VAL(vbuf[k]);
            Move m = VM_MOVE(vbuf[k]);
            if (!qstail) {
                int real;
                int score = score_move(pos, m, val, gamma, depth, rd, root, guard, &real);
                PROCESS(real, m, score);
            } else {
                Pos np = domove(pos, m);
                PROCESS(1, m, -bound(&np, 1 - gamma, rd - 1, 0, 0));
            }
            if (done) break;
        }
    }

after_moves:
    if (depth && !live) {
        if (!QS_TAIL) {
            /* Classify mate/stalemate exactly. */
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
        } else {
            /* PR #171: find the first legal move lazily; mate/stalemate
             * only when none exists; if every legal move sits below this
             * node's QS threshold, retry just the filtered tail (once --
             * never from inside a tail probe).  Lazy short-circuit order
             * matches the PR: the scan stops at the first legal move at
             * or above the threshold. */
            struct qtctx tc = { pos, 0, 0, val_lower, qstail };
            gen_moves(pos, qt_cb, &tc);
            if (!tc.found) {
                Pos rp = rotate(pos, 1);
                if (king_capture(&rp, NULL))
                    best = MATE_DIST
                         ? imax(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)
                         : -MATE_LOWER;
                else
                    best = 0;
            } else if (!qstail && !tc.above) {
                int s = bound(pos, gamma, depth, 1, 1);
                if (s > best) best = s;
            }
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
    gen_calls = 0;
    map_clear(&tps);
    PSTP['K'] = (memchr(hist[nhist - 1].b, 'Q', 120) && memchr(hist[nhist - 1].b, 'q', 120))
              ? TAB[5] : KEND;
    if (DERIVE_FRESH) {
        /* PR #184: history[:] = [p._replace(score=evaluate(p.board))] --
         * every score re-derived from the board under the K-table chosen
         * for THIS search; no score is inherited across table swaps. */
        for (int k = 0; k < nhist; k++) {
            int score = 0;
            for (int i = 0; i < 120; i++) {
                char c = hist[k].b[i];
                if (isup(c)) score += PSTP[(int)c][i];
                else if (islo(c)) score -= PSTP[c - 32][119 - i];
            }
            hist[k].score = score;
            pos_seal(&hist[k]);
        }
    }
    rootpos = hist[nhist - 1];
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
            int score = bound(&rootpos, gamma, depth, 1, 0);
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
    printf("done nodes %ld gen %ld\n", last_nodes, gen_calls);
    fflush(stdout);
}

/* Game-loop go for surrogate matches: fixed nodes (primary), or movetime.
 * The consumer is a transcription of sunfish_ui/uci.py's go_loop, which is
 * what pypy-classic plays through: candidates from fail-highs, committed
 * when their depth completes; the node cap checked BETWEEN probes, AFTER
 * the yield is consumed, and only at depth > 1 -- the probe that crosses
 * the cap always finishes and its result counts (both sides overshoot by
 * at most one MTD probe; a mid-probe abort here made the twin play one
 * depth staler than pypy whenever the cap landed inside the first, long,
 * probe of a new depth: the calibration match caught it at -63 Elo).
 * Upperbound probes are reported too, so draw/resign adjudication sees
 * the same score stream both sides.  bestmove falls back to the first
 * legal move (never "(none)" while one exists), like uci.py's floor. */
static void render_sq(char *buf, int i) {
    buf[0] = (char)('a' + pymod(i - A1, 10));
    buf[1] = (char)('0' + (1 - pyfloordiv(i - A1, 10)));
}
struct floorctx { const Pos *p; int found; Move m; };
static int floor_cb(Move m, void *vc) {
    struct floorctx *c = vc;
    Pos child = domove(c->p, m);
    if (king_capture(&child, NULL)) return 0;
    c->m = m; c->found = 1; return 1;
}
static void go_game(long max_nodes, double movetime_s, int maxd) {
    search_setup();
    node_cap = 0;                    /* nodes: yield-boundary rule only */
    double start = now_s();
    deadline = movetime_s > 0 ? start + (movetime_s > 0.05 ? movetime_s : 0.05) : 0.0;
    char best[8] = "", cand[8] = "";
    int d0 = 1;
    int mover_black = mover_is_black();

    if (!setjmp(stopjmp)) {          /* movetime deadline still aborts mid-probe */
        int gamma = 0;
        for (int depth = 1; depth < 1000 && depth <= maxd; depth++) {
            int lower = 1 - MATE_UPPER, upper = MATE_UPPER;
            while (lower < upper - EVAL_ROUGHNESS) {
                int score = bound(&rootpos, gamma, depth, 1, 0);
                if (score >= gamma) lower = score;
                if (score < gamma) upper = score;
                /* --- yield consumer (uci.py go_loop) --- */
                if (depth > d0) {
                    if (cand[0]) strcpy(best, cand);
                    d0 = depth;
                }
                if (score >= gamma) {
                    Move mv;
                    if (!tpm_get(&rootpos, &mv)) {
                        /* root fail-high without a move: verified terminal,
                         * exact score; nothing to search or play. */
                        printf("info depth %d score cp %d nodes %ld\n",
                               depth, score, nodes);
                        goto out;
                    }
                    int i = mv.i, j = mv.j;
                    if (mover_black) { i = 119 - i; j = 119 - j; }
                    render_sq(cand, i); render_sq(cand + 2, j);
                    cand[4] = mv.prom ? mv.prom + 32 : 0;
                    cand[5] = 0;
                    printf("info depth %d score cp %d lowerbound nodes %ld pv %s\n",
                           depth, score, nodes, cand);
                } else {
                    printf("info depth %d score cp %d upperbound nodes %ld\n",
                           depth, score, nodes);
                }
                if (depth > 1) {
                    if (max_nodes && nodes >= max_nodes)
                        goto out;
                    if (deadline != 0.0
                            && now_s() - start > (deadline - start) * 0.8)
                        goto out;
                }
                gamma = pyfloordiv(lower + upper + 1, 2);
            }
        }
    }
out:
    node_cap = 0; deadline = 0.0;
    if (!best[0] && !cand[0]) {
        /* Structural bestmove floor: never "(none)" while a legal move
         * exists (uci.py's floor; the recent master rule). */
        struct floorctx fc = { &rootpos, 0, { 0, 0, 0 } };
        gen_moves(&rootpos, floor_cb, &fc);
        if (fc.found) {
            int i = fc.m.i, j = fc.m.j;
            if (mover_black) { i = 119 - i; j = 119 - j; }
            render_sq(cand, i); render_sq(cand + 2, j);
            cand[4] = fc.m.prom ? fc.m.prom + 32 : 0;
            cand[5] = 0;
        }
    }
    printf("bestmove %s\n", best[0] ? best : cand[0] ? cand : "(none)");
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Position setup                                                      */
/* ------------------------------------------------------------------ */
static void reset_state(void) {
    map_clear(&tps);
    map_clear(&tpm);
    kslot_reset();
    nodes = 0;
    PSTP['K'] = TAB[5];
    memcpy(hist[0].b, INITIAL, 120);
    hist[0].score = 0;
    hist[0].wc0 = hist[0].wc1 = hist[0].bc0 = hist[0].bc1 = 1;
    hist[0].ep = hist[0].kp = 0;
    pos_seal(&hist[0]);
    nhist = nhist0 = 1;
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
    /* uci.py: hist = [pos] if WHITE else [pos.rotate(), pos], with pos already
     * oriented to the mover.  rotate() is an involution, so the driver's
     * hist[0] is the board exactly as parsed -- p here. */
    if (side0 == 'b') {
        if (FEN_HIST) { hist[0] = p; hist[1] = rotate(&p, 0); nhist = 2; }
        else          { hist[0] = rotate(&p, 0);              nhist = 1; }
    } else {
        hist[0] = p;
        nhist = 1;
    }
    nhist0 = nhist;
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
static int scale_eval(int value, int scale) {
    int product = value * scale;
    return (product + (product >= 0 ? 50 : -50)) / 100;
}

static void refresh_eval(void) {
    if (!tables_loaded) return;
    int values[] = {REF_PIECEVAL[0], VALUE_N, VALUE_B, VALUE_R, VALUE_Q,
                    REF_PIECEVAL[5]};
    int scales[] = {PST_P, PST_N, PST_B, PST_R, PST_Q, PST_K};
    for (int p = 0; p < 6; p++) {
        PIECEVAL[p] = values[p];
        for (int i = 0; i < 120; i++)
            TAB[p][i] = values[p] + scale_eval(
                REF_TAB[p][i] - REF_PIECEVAL[p], scales[p]);
    }
    for (int i = 0; i < 120; i++)
        KEND[i] = PIECEVAL[5] + scale_eval(
            REF_KEND[i] - REF_PIECEVAL[5], PST_KE);
    MATE_LOWER = PIECEVAL[5] - 13 * PIECEVAL[4];
    MATE_UPPER = PIECEVAL[5] + 10 * PIECEVAL[4];
}

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
    memcpy(REF_TAB, TAB, sizeof TAB);
    memcpy(REF_KEND, KEND, sizeof KEND);
    memcpy(REF_PIECEVAL, PIECEVAL, sizeof PIECEVAL);
    tables_loaded = 1;
    refresh_eval();
    return 1;
}

struct knob { const char *name; int *ip; long *lp; };
static struct knob KNOBS[] = {
    { "QS", &QS, NULL }, { "QS_A", &QS_A, NULL }, { "LMR", &LMR, NULL },
    { "EVAL_ROUGHNESS", &EVAL_ROUGHNESS, NULL },
    { "NULL_CAP_MARGIN", &NULL_CAP_MARGIN, NULL },
    { "VALUE_N", &VALUE_N, NULL }, { "VALUE_B", &VALUE_B, NULL },
    { "VALUE_R", &VALUE_R, NULL }, { "VALUE_Q", &VALUE_Q, NULL },
    { "PST_P", &PST_P, NULL }, { "PST_N", &PST_N, NULL },
    { "PST_B", &PST_B, NULL }, { "PST_R", &PST_R, NULL },
    { "PST_Q", &PST_Q, NULL }, { "PST_K", &PST_K, NULL },
    { "PST_KE", &PST_KE, NULL },
    { "TABLE_SIZE", NULL, &TABLE_SIZE },
    { "NULL_MARGIN", &NULL_MARGIN, NULL },
    { "NULL_MIN_DEPTH", &NULL_MIN_DEPTH, NULL },
    { "NULL_LIMIT", &NULL_LIMIT, NULL },
    { "NULL_CUT_RED", &NULL_CUT_RED, NULL },
    { "NULL_RED", &NULL_RED, NULL },
    { "IID_MIN_DEPTH", &IID_MIN_DEPTH, NULL },
    { "IID_RED", &IID_RED, NULL },
    { "FUT_MAX", &FUT_MAX, NULL },
    { "FUT_CAP", &FUT_CAP, NULL },
    { "FUT_CAP_DEPTH", &FUT_CAP_DEPTH, NULL },
    { "MATE_DIST", &MATE_DIST, NULL },
    { "EVICT_POLICY", &EVICT_POLICY, NULL },
    { "EVICT_SCAN_K", &EVICT_SCAN_K, NULL },
    { "KILLER_COUNT", &KILLER_COUNT, NULL },
    { "USE_VARIANT", &USE_VARIANT, NULL },
    { "QS_TAIL", &QS_TAIL, NULL },
    { "FUEL_NULL", &FUEL_NULL, NULL },
    { "FUEL_MIN_DEPTH", &FUEL_MIN_DEPTH, NULL },
    { "DERIVE_FRESH", &DERIVE_FRESH, NULL },
    { "FEN_HIST", &FEN_HIST, NULL },
    { NULL, NULL, NULL }
};
static int set_knob(const char *name, long v) {
    /* Out-of-range battery knobs are a hard error, not a clamp: a
     * silently adjusted knob would fake a variant. */
    if (!strcmp(name, "EVICT_POLICY") && (v < 0 || v > 3)) return 0;
    if (!strcmp(name, "EVICT_SCAN_K") && v < 1) return 0;
    if (!strcmp(name, "KILLER_COUNT") && (v < 1 || v > MAXKILL)) return 0;
    if (!strcmp(name, "QS_TAIL") && (v < 0 || v > 1)) return 0;
    if (!strcmp(name, "FUEL_NULL") && (v < 0 || v > 2)) return 0;
    if (!strcmp(name, "FUT_CAP") && (v < 0 || v > 2)) return 0;
    if (!strcmp(name, "FUT_CAP_DEPTH") && (v < 2 || v > 6)) return 0;
    if (!strcmp(name, "FUEL_MIN_DEPTH") && v < 1) return 0;
    if (!strcmp(name, "DERIVE_FRESH") && (v < 0 || v > 1)) return 0;
    if (!strcmp(name, "FEN_HIST") && (v < 0 || v > 1)) return 0;
    for (struct knob *k = KNOBS; k->name; k++)
        if (!strcmp(k->name, name)) {
            if (k->ip) *k->ip = (int)v; else *k->lp = v;
            refresh_eval();
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
    /* argv knobs after the table path: NAME=VALUE (for match harnesses
     * that can pass args but not stdin preambles).  Unknown names are a
     * hard error -- a silently ignored knob would fake a variant. */
    for (int a = 2; a < argc; a++) {
        char nm[64];
        const char *eq = strchr(argv[a], '=');
        if (!eq || eq - argv[a] >= (long)sizeof nm) {
            fprintf(stderr, "ctwin: bad knob arg %s\n", argv[a]);
            return 1;
        }
        memcpy(nm, argv[a], eq - argv[a]);
        nm[eq - argv[a]] = 0;
        if (!set_knob(nm, atol(eq + 1))) {
            fprintf(stderr, "ctwin: unknown knob %s\n", nm);
            return 1;
        }
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

        else if (!strcmp(tok[0], "setoption")) {
            if (ntok < 5 || strcmp(tok[1], "name") || strcmp(tok[3], "value")
                    || !set_knob(tok[2], atol(tok[4])))
                fprintf(stderr, "ctwin: bad UCI option\n");
        }

        else if (!strcmp(tok[0], "position")) {
            if (ntok >= 2 && !strcmp(tok[1], "startpos")) {
                reset_ok:
                memcpy(hist[0].b, INITIAL, 120);
                hist[0].score = 0;
                hist[0].wc0 = hist[0].wc1 = hist[0].bc0 = hist[0].bc1 = 1;
                hist[0].ep = hist[0].kp = 0;
                pos_seal(&hist[0]);
                nhist = nhist0 = 1;
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
            if (nhist > nhist0) nhist--;
            puts("ok");
        }

        else if (!strcmp(tok[0], "moves")) {
            struct listctx c = { &hist[nhist - 1] };
            gen_moves(&hist[nhist - 1], list_cb, &c);
            puts("end");
        }

        else if (!strcmp(tok[0], "go")) {
            long gnodes = 0, wtime = -1, btime = -1, winc = 0, binc = 0;
            double mt = 0;
            int gdepth = 0, movestogo = 0;
            for (int k = 1; k + 1 < ntok; k += 2) {
                if (!strcmp(tok[k], "depth")) gdepth = atoi(tok[k + 1]);
                else if (!strcmp(tok[k], "nodes")) gnodes = atol(tok[k + 1]);
                else if (!strcmp(tok[k], "movetime")) mt = atol(tok[k + 1]) / 1000.0;
                else if (!strcmp(tok[k], "wtime")) wtime = atol(tok[k + 1]);
                else if (!strcmp(tok[k], "btime")) btime = atol(tok[k + 1]);
                else if (!strcmp(tok[k], "winc")) winc = atol(tok[k + 1]);
                else if (!strcmp(tok[k], "binc")) binc = atol(tok[k + 1]);
                else if (!strcmp(tok[k], "movestogo")) movestogo = atoi(tok[k + 1]);
            }
            if (mt == 0 && (wtime >= 0 || btime >= 0)) {
                int black = mover_is_black();
                double remain = (black ? btime : wtime) / 1000.0;
                double inc = (black ? binc : winc) / 1000.0;
                mt = movestogo ? remain / movestogo + inc : remain / 12 + 0.9 * inc;
                double cap = remain / 2 - 1;
                if (cap < mt) mt = cap;
                if (mt < 0.05) mt = 0.05;
            }
            if (gdepth && !gnodes && mt == 0) go_depth(gdepth);
            else go_game(gnodes, mt, gdepth ? gdepth : 999);
        }

        else if (!strcmp(tok[0], "uci")) {
            puts("id name sunfish ctwin");
            for (struct knob *k = KNOBS; k->name; k++) {
                long value = k->ip ? *k->ip : *k->lp;
                printf("option name %s type spin default %ld min -1000000000 max 1000000000\n",
                       k->name, value);
            }
            puts("uciok");
        }
        else if (!strcmp(tok[0], "isready")) puts("readyok");
        else printf("err unknown command: %s\n", tok[0]);
        fflush(stdout);
    }
    return 0;
}
