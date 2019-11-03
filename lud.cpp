#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "sys/time.h"

#define min(x, y) ((x) < (y) ? (x) : (y))

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <seed_number>\n", argv[0]);
        return -1;
    }

    struct timeval start;
    struct timeval end;

    int n = atoi(argv[1]);
    int s = atoi(argv[2]);

    double *matrixA = new double[n * n];
    double *matrixLU = new double[n * n];
    double *matrixB = new double[n * n];

    // Build Matrix
    srand(s);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrixA[i * n + j] = matrixLU[i * n + j] = (double)rand();
        }
    }

    // LU Decomposition
    gettimeofday(&start, NULL);
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            matrixLU[i * n + k] /= matrixLU[k * n + k];
        }
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                matrixLU[i * n + j] -= matrixLU[i * n + k] * matrixLU[k * n + j];
            }
        }
    }
    gettimeofday(&end, NULL);
    printf("LU Decomposition: %f s\n",(end.tv_sec-start.tv_sec)+(float)(end.tv_usec-start.tv_usec) / 1000000.0);

    // Reconstruct Matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int min_idx = min(i - 1, j);
            matrixB[i * n + j] = (j >= i ? matrixLU[i * n + j] : 0);
            for (int k = 0; k <= min_idx; k++) {
                matrixB[i * n + j] += matrixLU[i * n + k] * matrixLU[k * n + j];
            }
            matrixB[i * n + j] = matrixB[i * n + j];
        }
    }

    // Calculate Matrix Difference
    double diff = 0;
    for (int j = 0; j < n; j++) {
        double row_sum = 0;
        for (int i = 0; i < n; i++) {
            row_sum += (matrixA[i * n + j] - matrixB[i * n + j]) * (matrixA[i * n + j] - matrixB[i * n + j]);
        }
        diff += sqrt(row_sum);
    }
    printf("Sum of difference: %.10lf\n", diff);

    // Memory Deallocation
    delete [] matrixA;
    delete [] matrixLU;
    delete [] matrixB;

    return 0;
}
