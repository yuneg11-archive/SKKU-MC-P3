#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include "sys/time.h"

#define min(x, y) ((x) < (y) ? (x) : (y))

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <seed_number>\n", argv[0]);
        return -1;
    }

    struct timeval start;
    struct timeval end;

    int mat_len = atoi(argv[1]);
    int seed = atoi(argv[2]);

    double *matrixA = new double[mat_len * mat_len];
    double *matrixLU = new double[mat_len * mat_len];
    double *matrixB = new double[mat_len * mat_len];

    if (MPI_Init(NULL, NULL) != 0) {
        printf("Error: MPI_Init failed\n");
        return -1;
    }

    // MPI Region
    {
        // Initialize MPI Communicator
        int rank, comm_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        int comm_per_line = (int)sqrt(comm_size);
        MPI_Comm *row_comm = new MPI_Comm[comm_per_line-1];
        MPI_Comm *col_comm = new MPI_Comm[comm_per_line-1];
        int *row_rank = new int[comm_per_line-1];
        int *col_rank = new int[comm_per_line-1];
        int row_order = rank / comm_per_line;
        int col_order = rank % comm_per_line;
        int calc_order = (row_order < col_order ? row_order : col_order);

        for (int i = 0; i < comm_per_line-1; i++) {
            row_rank[i] = (col_order >= i && row_order >= i ? row_order - i : -1);
            col_rank[i] = (col_order >= i && row_order >= i ? col_order - i : -1);
        }

        MPI_Comm_split(MPI_COMM_WORLD, rank % comm_per_line, rank, &row_comm[0]);
        MPI_Comm_split(MPI_COMM_WORLD, rank / comm_per_line, rank, &col_comm[0]);
        for (int i = 1; i <= calc_order+1 && i < comm_per_line-1; i++) {
            MPI_Comm_split(row_comm[i-1], (row_rank[i-1] == 0 ? 0 : 1), row_rank[i-1], &row_comm[i]);
            MPI_Comm_split(col_comm[i-1], (col_rank[i-1] == 0 ? 0 : 1), col_rank[i-1], &col_comm[i]);
        }

        // Build Matrix
        srand(seed);
        for (int i = 0; i < mat_len; i++) {
            for (int j = 0; j < mat_len; j++) {
                matrixA[i * mat_len + j] = matrixLU[i * mat_len + j] = (double)rand();
            }
        }

        // LU Decomposition
        gettimeofday(&start, NULL);

        for (int k = 0; k < mat_len; k++) {
            for (int i = k + 1; i < mat_len; i++) {
                matrixLU[i * mat_len + k] /= matrixLU[k * mat_len + k];
            }
            for (int i = k + 1; i < mat_len; i++) {
                for (int j = k + 1; j < mat_len; j++) {
                    matrixLU[i * mat_len + j] -= matrixLU[i * mat_len + k] * matrixLU[k * mat_len + j];
                }
            }
        }

        gettimeofday(&end, NULL);
        printf("LU Decomposition: %f s\n",(end.tv_sec-start.tv_sec)+(float)(end.tv_usec-start.tv_usec) / 1000000.0);

        // Reconstruct Matrix
        for (int i = 0; i < mat_len; i++) {
            for (int j = 0; j < mat_len; j++) {
                int min_idx = min(i - 1, j);
                matrixB[i * mat_len + j] = (j >= i ? matrixLU[i * mat_len + j] : 0);
                for (int k = 0; k <= min_idx; k++) {
                    matrixB[i * mat_len + j] += matrixLU[i * mat_len + k] * matrixLU[k * mat_len + j];
                }
                matrixB[i * mat_len + j] = matrixB[i * mat_len + j];
            }
        }

        // Calculate Matrix Difference
        double diff = 0;
        for (int j = 0; j < mat_len; j++) {
            double row_sum = 0;
            for (int i = 0; i < mat_len; i++) {
                row_sum += (matrixA[i * mat_len + j] - matrixB[i * mat_len + j]) * (matrixA[i * mat_len + j] - matrixB[i * mat_len + j]);
            }
            diff += sqrt(row_sum);
        }
        printf("Sum of difference: %.10lf\n", diff);
    }

    if (MPI_Finalize() != 0) {
        printf("Error: MPI_Finalize failed\n");
        return -1;
    }

    // Memory Deallocation
    delete [] matrixA;
    delete [] matrixLU;
    delete [] matrixB;

    return 0;
}
