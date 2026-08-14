# ctwin: C twin of classic sunfish.py, node-identity lab instrument.
CC ?= cc
CFLAGS ?= -O3 -march=native -Wall -Wextra

all: sunfish_c tables_classic.txt

sunfish_c: sunfish.c
	$(CC) $(CFLAGS) -o $@ $<

tables_classic.txt: gen_tables.py ../../sunfish.py
	python3 gen_tables.py > $@

# Quick differential probe (seconds): movegen + search identity on a
# small position set.  `make test-full` is the wide sweep.
test: all
	python3 difftest.py --quick

test-full: all
	python3 difftest.py --n 40 --depth 6

# The FULL identity gate: what "difftest passes" means in TESTING.md and
# README.md.  Every change to sunfish.c or to the reference must re-pass
# this in full -- identity is a regression gate, never a one-time claim.
gate: all
	python3 difftest.py --n 40 --depth 6 --walk
	python3 difftest.py --n 5 --depth 7
	python3 difftest.py --n 40 --depth 5 --set QS=100 --set EVAL_ROUGHNESS=5
	python3 difftest.py --n 40 --depth 5 --set QS=0 --set EVAL_ROUGHNESS=40
	python3 difftest.py --n 40 --depth 6 --set TABLE_SIZE=500
	python3 difftest.py --n 40 --depth 6 --set TABLE_SIZE=50

bench: all
	python3 difftest.py --bench --depth 7

# Profile-guided rebuild of sunfish_c in place (clang toolchain; on
# Linux replace `xcrun llvm-profdata` with `llvm-profdata`).  Training
# run: a deep default-knob search.  PGO changes scheduling, never
# semantics -- but the gate still reruns on the PGO binary, because the
# binary that ships numbers is the binary that passed.
pgo: all
	$(CC) $(CFLAGS) -fprofile-instr-generate -o sunfish_c_pgen sunfish.c
	printf 'position startpos\ngo depth 9\nquit\n' | \
	    LLVM_PROFILE_FILE=ctwin.profraw ./sunfish_c_pgen tables_classic.txt > /dev/null
	xcrun llvm-profdata merge -output=ctwin.profdata ctwin.profraw
	$(CC) $(CFLAGS) -fprofile-instr-use=ctwin.profdata -o sunfish_c sunfish.c
	rm -f sunfish_c_pgen ctwin.profraw ctwin.profdata

clean:
	rm -f sunfish_c sunfish_c_pgen ctwin.profraw ctwin.profdata

.PHONY: all test test-full bench clean
