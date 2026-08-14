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

bench: all
	python3 difftest.py --bench

clean:
	rm -f sunfish_c

.PHONY: all test test-full bench clean
