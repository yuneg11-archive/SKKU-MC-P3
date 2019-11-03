#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
    int n, s;
    double *matrix;

    if (argc != 3) {
        printf("Usage: %s <matrix_size> <seed_number>\n", argv[0]);
        return -1;
    } else {
        n = atoi(argv[1]);
        s = atoi(argv[2]);
        printf("Matrix size: %d x %d\n", n, n);
        printf("Random seed: %d\n", s);
    }

    // Memory Allocation
    matrix = new double[n * n];

    // Build Matrix
    srand(s);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (double)rand();
        }
    }

    // Debug Matrix Print
    if (true) {
        for (int i = 0; i < n; i++) {
            printf("[");
            for (int j = 0; j < n; j++) {
                printf(" %14.3lf", matrix[i * n + j]);
            }
            printf("]\n");
        }
    }

    // Memory Deallocation
    delete [] matrix;

    return 0;
}
