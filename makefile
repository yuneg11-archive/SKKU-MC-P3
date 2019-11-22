CC=g++
MPICC=mpic++
FFLAGS=-Wall -Wno-reorder

all: seq par

seq: lud.cpp
	$(CC) $(FFLAGS) lud.cpp -o lud_seq

par: lud2.cpp
	$(MPICC) $(FFLAGS) lud2.cpp -o lud_par

clean:
	rm -f lud_seq lud_par

