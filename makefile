CC=g++
MPICC=mpic++
FFLAGS=-g -Wall

all: seq par

seq: lud.cpp
	$(CC) $(FFLAGS) lud.cpp -o lud_seq

par: lud2.cpp
	$(MPICC) $(FFLAGS) lud2.cpp -o lud_par

verify: verify.cpp
	$(CC) $(FFLAGS) verify.cpp -o verify

clean:
	rm -f lud_seq lud_par verify

