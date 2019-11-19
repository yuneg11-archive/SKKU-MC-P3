#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

#define min(x, y) ((x) < (y) ? (x) : (y))

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <seed_number>\n", argv[0]);
        return -1;
    }

    int mat_len = atoi(argv[1]);
    int seed = atoi(argv[2]);

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

        // Allocation
        int sub_mat_len = mat_len / comm_per_line + (mat_len % comm_per_line == 0 ? 0 : 1);
        int sub_mat_row_len = sub_mat_len - (row_rank[0] == comm_per_line-1 ? comm_per_line - mat_len % comm_per_line : 0);
        int sub_mat_col_len = sub_mat_len - (col_rank[0] == comm_per_line-1 ? comm_per_line - mat_len % comm_per_line : 0);
        int sub_mat_row_base = sub_mat_len * row_order;
        int sub_mat_col_base = sub_mat_len * col_order;
        int sub_mat_l_row_len = sub_mat_len;
        int sub_mat_l_col_len = sub_mat_col_len;
        int sub_mat_u_row_len = sub_mat_row_len;
        int sub_mat_u_col_len = sub_mat_len;

        double *sub_mat = new double[sub_mat_len * sub_mat_len];
        double *sub_mat_lu = new double[sub_mat_len * sub_mat_len];
        double *sub_mat_l = new double[sub_mat_l_row_len * sub_mat_l_col_len];
        double *sub_mat_u = new double[sub_mat_u_row_len * sub_mat_u_col_len];
        double *sub_mat_recon = new double[sub_mat_len * sub_mat_len];

        // Build Matrix
        srand(seed);
        for (int row = 0; row < sub_mat_row_base; row++) {
            for (int col = 0; col < mat_len; col++) {
                rand();
            }
        }
        for (int row = 0; row < sub_mat_row_len; row++) {
            for (int col = 0; col < sub_mat_col_base; col++) {
                rand();
            }
            for (int col = 0; col < sub_mat_col_len; col++) {
                sub_mat[row * sub_mat_col_len + col] = sub_mat_lu[row * sub_mat_col_len + col] = (double)rand();
            }
            for (int col = sub_mat_col_base + sub_mat_col_len; col < mat_len; col++) {
                rand();
            }
        }

        // LU Decomposition
        for (int step = 0; step <= calc_order; step++) {
            if (row_rank[step] == 0 && col_rank[step] == 0) {
                for (int k = 0; k < sub_mat_row_len; k++) {
                    for (int i = k + 1; i < sub_mat_row_len; i++) {
                        sub_mat_lu[i * sub_mat_col_len + k] /= sub_mat_lu[k * sub_mat_col_len + k];
                    }
                    for (int i = k + 1; i < sub_mat_row_len; i++) {
                        for (int j = k + 1; j < sub_mat_col_len; j++) {
                            sub_mat_lu[i * sub_mat_col_len + j] -= sub_mat_lu[i * sub_mat_col_len + k] * sub_mat_lu[k * sub_mat_col_len + j];
                        }
                    }
                }
                if (step < comm_per_line-1) {
                    printf("%2d (%d, %d) - L Send %d\n", rank, row_rank[0], col_rank[0], step);
                    printf("%2d (%d, %d) - U Send %d\n", rank, row_rank[0], col_rank[0], step);
                    MPI_Bcast(sub_mat_lu, sub_mat_row_len * sub_mat_col_len, MPI_DOUBLE, 0, col_comm[step]); // Send to horizontal ones
                    MPI_Bcast(sub_mat_lu, sub_mat_row_len * sub_mat_col_len, MPI_DOUBLE, 0, row_comm[step]); // Send to vertical ones
                }
            } else if (row_rank[step] == 0) { // Horizontal ones
                MPI_Bcast(sub_mat_l, sub_mat_l_row_len * sub_mat_l_col_len, MPI_DOUBLE, 0, col_comm[step]);
                printf("%2d (%d, %d) - L Recv %d\n", rank, row_rank[0], col_rank[0], step);
            } else if (col_rank[step] == 0) { // Vertical ones
                MPI_Bcast(sub_mat_u, sub_mat_u_row_len * sub_mat_u_col_len, MPI_DOUBLE, 0, row_comm[step]);
                printf("%2d (%d, %d) - U Recv %d\n", rank, row_rank[0], col_rank[0], step);
            } else {

            }
        }

        /*
        // LU Decomposition
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
        printf("%.10lf\n", diff);
        */
    }

    if (MPI_Finalize() != 0) {
        printf("Error: MPI_Finalize failed\n");
        return -1;
    }

    return 0;
}
