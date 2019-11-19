#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <mpi.h>

#define min(x, y) ((x) < (y) ? (x) : (y))

void lu_decompose(double *mat, int len) {
    for (int k = 0; k < len; k++) {
        for (int i = k + 1; i < len; i++) {
            mat[i * len + k] /= mat[k * len + k];
        }
        for (int i = k + 1; i < len; i++) {
            for (int j = k + 1; j < len; j++) {
                mat[i * len + j] -= mat[i * len + k] * mat[k * len + j];
            }
        }
    }
}

void l_inverse(double *mat, int len) {
    for (int i = 0; i < len; i++) {
        mat[i * len + i] = 1;
        for (int j = 0; j < i; j++) {
            double sum = 0;
            for (int k = 0; k < i; k++) {
                sum -= mat[i * len + k] * mat[k * len + j];
            }
            mat[i * len + j] = sum;
        }
        for (int j = i+1; j < len; j++) {
            mat[i * len + j] = 0;
        }
    }
}

void l_multiply(double *mat_u, double *mat_l, double *mat, int mat_l_len, int mat_col_len) {
    for (int i = 0; i < mat_l_len; i++) {
        for (int j = 0; j < mat_col_len; j++) {
            mat_u[i * mat_col_len + j] = 0;
            for (int k = 0; k <= i; k++) {
                mat_u[i * mat_col_len + j] += mat_l[i * mat_l_len + k] * mat[k * mat_col_len + j];
            }
        }
    }
}

void u_inverse(double *mat, int len) {
    for (int i = len-1; i >= 0; i--) {
        mat[i * len + i] = 1 / mat[i * len + i];
        for (int j = len-1; j > i; j--) {
            double sum = 0;
            for (int k = len-1; k > i; k--) {
                sum -= mat[i * len + k] * mat[k * len + j];
            }
            mat[i * len + j] = sum * mat[i * len + i];
        }
    }
}

void u_multiply(double *mat_l, double *mat, double *mat_u, int mat_row_len, int mat_u_len) {
    for (int i = 0; i < mat_row_len; i++) {
        for (int j = 0; j < mat_u_len; j++) {
            mat_l[i * mat_u_len + j] = 0;
            for (int k = 0; k <= j; k++) {
                mat_l[i * mat_u_len + j] += mat[i * mat_u_len + k] * mat_u[k * mat_u_len + j];
            }
        }
    }
}

void matrix_multiply(double *mat_out, double *mat1, double *mat2, int mat_row_len, int mat_inter_len, int mat_col_len, int mode) {
    for (int i = 0; i < mat_row_len; i++) {
        for (int j = 0; j < mat_col_len; j++) {
            double sum = 0;
            for (int k = 0; k < mat_inter_len; k++) {
                sum += mat1[i * mat_inter_len + k] * mat2[k * mat_col_len + j];
            }
            if (mode == -1) {
                mat_out[i * mat_col_len + j] -= sum;
            } else if (mode == 1) {
                mat_out[i * mat_col_len + j] += sum;
            } else {
                mat_out[i * mat_col_len + j] = sum;
            }
        }
    }
}

void lu_extract(double *mat_l, double *mat_u, double *mat_lu, int mat_len, int mat_row_len, int mat_col_len) {
    for (int i = 0; i < mat_row_len; i++) {
        for (int j = 0; j < i; j++) {
            mat_l[i * mat_len + j] = mat_lu[i * mat_col_len + j];
            mat_u[i * mat_col_len + j] = 0;
        }
        mat_l[i * mat_len + i] = 1;
        mat_u[i * mat_len + i] = mat_lu[i * mat_col_len + i];
        for (int j = i+1; j < mat_col_len; j++) {
            mat_l[i * mat_len + j] = 0;
            mat_u[i * mat_col_len + j] = mat_lu[i * mat_col_len + j];
        }
        for (int j = mat_col_len; j < mat_len; j++) {
            mat_l[i * mat_len + j] = 0;
        }
    }
    for (int i = mat_row_len; i < mat_len; i++) {
        for (int j = 0; j < mat_len; j++) {
            mat_u[i * mat_col_len + j] = 0;
        }
    }
}

void l_extract(double *mat_l, double *mat_lu, int mat_len, int mat_row_len) {
    for (int i = 0; i < mat_row_len; i++) {
        for (int j = 0; j < mat_len; j++) {
            mat_l[i * mat_len + j] = mat_lu[i * mat_len + j];
        }
    }
}

void u_extract(double *mat_u, double *mat_lu, int mat_len, int mat_col_len) {
    for (int i = 0; i < mat_len; i++) {
        for (int j = 0; j < mat_col_len; j++) {
            mat_u[i * mat_col_len + j] = mat_lu[i * mat_col_len + j];
        }
    }
}

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
            row_rank[i] = (col_order >= i && row_order >= i ? col_order - i : -1);
            col_rank[i] = (col_order >= i && row_order >= i ? row_order - i : -1);
        }

        MPI_Comm_split(MPI_COMM_WORLD, rank / comm_per_line, rank, &row_comm[0]);
        MPI_Comm_split(MPI_COMM_WORLD, rank % comm_per_line, rank, &col_comm[0]);
        for (int i = 1; i <= calc_order+1 && i < comm_per_line-1; i++) {
            MPI_Comm_split(row_comm[i-1], (row_rank[i-1] == 0 ? 0 : 1), row_rank[i-1], &row_comm[i]);
            MPI_Comm_split(col_comm[i-1], (col_rank[i-1] == 0 ? 0 : 1), col_rank[i-1], &col_comm[i]);
        }

        // Allocation
        int sub_mat_len = mat_len / comm_per_line + (mat_len % comm_per_line == 0 ? 0 : 1);
        int sub_mat_row_len = sub_mat_len - (row_order == comm_per_line-1 ? comm_per_line - mat_len % comm_per_line : 0);
        int sub_mat_col_len = sub_mat_len - (col_order == comm_per_line-1 ? comm_per_line - mat_len % comm_per_line : 0);
        int sub_mat_row_base = sub_mat_len * row_order;
        int sub_mat_col_base = sub_mat_len * col_order;
        int sub_mat_l_row_len = sub_mat_row_len;
        int sub_mat_l_col_len = sub_mat_len;
        int sub_mat_u_row_len = sub_mat_len;
        int sub_mat_u_col_len = sub_mat_col_len;

        double *sub_mat = new double[sub_mat_row_len * sub_mat_col_len];
        double *sub_mat_lu = new double[sub_mat_row_len * sub_mat_col_len];
        double *sub_mat_l = new double[sub_mat_l_row_len * sub_mat_l_col_len];
        double *sub_mat_u = new double[sub_mat_u_row_len * sub_mat_u_col_len];

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
                lu_decompose(sub_mat_lu, sub_mat_row_len);
                if (step < comm_per_line-1) {
                    MPI_Bcast(sub_mat_lu, sub_mat_row_len * sub_mat_col_len, MPI_DOUBLE, 0, row_comm[step]); // Send L
                    MPI_Bcast(sub_mat_lu, sub_mat_row_len * sub_mat_col_len, MPI_DOUBLE, 0, col_comm[step]); // Send U
                }
                lu_extract(sub_mat_l, sub_mat_u, sub_mat_lu, sub_mat_len, sub_mat_row_len, sub_mat_col_len);
            } else if (col_rank[step] == 0) {
                MPI_Bcast(sub_mat_l, sub_mat_l_row_len * sub_mat_l_col_len, MPI_DOUBLE, 0, row_comm[step]); // Receive L
                l_inverse(sub_mat_l, sub_mat_l_row_len);
                l_multiply(sub_mat_lu, sub_mat_l, sub_mat, sub_mat_l_row_len, sub_mat_col_len);
                MPI_Bcast(sub_mat_lu, sub_mat_row_len * sub_mat_col_len, MPI_DOUBLE, 0, col_comm[step]); // Send U
                u_extract(sub_mat_u, sub_mat_lu, sub_mat_len, sub_mat_col_len);
            } else if (row_rank[step] == 0) {
                MPI_Bcast(sub_mat_u, sub_mat_u_row_len * sub_mat_u_col_len, MPI_DOUBLE, 0, col_comm[step]); // Receive U
                u_inverse(sub_mat_u, sub_mat_u_row_len);
                u_multiply(sub_mat_lu, sub_mat, sub_mat_u, sub_mat_row_len, sub_mat_u_col_len);
                MPI_Bcast(sub_mat_lu, sub_mat_row_len * sub_mat_col_len, MPI_DOUBLE, 0, row_comm[step]); // Send L
                l_extract(sub_mat_l, sub_mat_lu, sub_mat_len, sub_mat_row_len);
            } else {
                MPI_Bcast(sub_mat_l, sub_mat_l_row_len * sub_mat_l_col_len, MPI_DOUBLE, 0, row_comm[step]); // Receive L
                MPI_Bcast(sub_mat_u, sub_mat_u_row_len * sub_mat_u_col_len, MPI_DOUBLE, 0, col_comm[step]); // Receive U
                matrix_multiply(sub_mat_lu, sub_mat_l, sub_mat_u, sub_mat_row_len, sub_mat_len, sub_mat_col_len, -1);
            }
        }

        // Reconstruct Matrix
        double *sub_mat_recon = new double[sub_mat_len * sub_mat_len](); // Initialize with 0
        double *sub_mat_l_buf = new double[sub_mat_l_row_len * sub_mat_l_col_len];
        double *sub_mat_u_buf = new double[sub_mat_u_row_len * sub_mat_u_col_len];
        MPI_Request req[4];
        bool req_valid[4] = {false, false, false, false};
        if (row_order >= col_order) {
            if (row_order < comm_per_line-1) {
                MPI_Isend(sub_mat_l, sub_mat_l_row_len * sub_mat_l_col_len, MPI_DOUBLE, comm_per_line - row_order + col_order - 1, 0, row_comm[0], &req[0]);  // Send L
                req_valid[0] = true;
            } else {
                double *temp = sub_mat_l_buf; sub_mat_l_buf = sub_mat_l; sub_mat_l = temp;
            }
        }
        if (row_order <= col_order) {
            if (col_order < comm_per_line-1) {
                MPI_Isend(sub_mat_u, sub_mat_u_row_len * sub_mat_u_col_len, MPI_DOUBLE, comm_per_line - col_order + row_order - 1, 0, col_comm[0], &req[1]); // Send U
                req_valid[1] = true;
            } else {
                double *temp = sub_mat_u_buf; sub_mat_u_buf = sub_mat_u; sub_mat_u = temp;
            }
        }
        if (col_order >= comm_per_line - row_order - 1) {
            if (row_order < comm_per_line-1) {
                MPI_Irecv(sub_mat_l_buf, sub_mat_l_row_len * sub_mat_l_col_len, MPI_DOUBLE, - comm_per_line + row_order + col_order + 1, 0, row_comm[0], &req[2]); // Receive L
                req_valid[2] = true;
            }
            if (col_order < comm_per_line-1) {
                MPI_Irecv(sub_mat_u_buf, sub_mat_u_row_len * sub_mat_u_col_len, MPI_DOUBLE, - comm_per_line + row_order + col_order + 1, 0, col_comm[0], &req[3]); // Receive U
                req_valid[3] = true;
            }
        }
        for (int i = 0; i < 4; i++) {
            if (req_valid[i] == true) {
                MPI_Wait(&req[i], MPI_STATUS_IGNORE);
                req_valid[i] = false;
            }
        }

        int l_dest = (col_order + 1) % comm_per_line;
        int l_src = (col_order - 1 < 0 ? comm_per_line - 1 : col_order - 1);
        int u_dest = (row_order + 1) % comm_per_line;
        int u_src = (row_order - 1 < 0 ? comm_per_line - 1 : row_order - 1);
        for (int step = 0; step < comm_per_line; step++) {
            bool row_activate = (step + (step < col_order + 1 ? comm_per_line : 0) <= row_order + col_order + 1);
            bool col_activate = (step + (step < row_order + 1 ? comm_per_line : 0) <= row_order + col_order + 1);
            bool next_row_activate = (step+1 + (step+1 < col_order + 1 ? comm_per_line : 0) <= row_order + col_order + 1);
            bool next_col_activate = (step+1 + (step+1 < row_order + 1 ? comm_per_line : 0) <= row_order + col_order + 1);
            bool calc_activate = row_activate && col_activate;

            double *temp_l = sub_mat_l_buf; sub_mat_l_buf = sub_mat_l; sub_mat_l = temp_l;
            double *temp_u = sub_mat_u_buf; sub_mat_u_buf = sub_mat_u; sub_mat_u = temp_u;

            if (row_activate == true && step < comm_per_line-1) {
                MPI_Isend(sub_mat_l, sub_mat_l_row_len * sub_mat_l_col_len, MPI_DOUBLE, l_dest, step+1, row_comm[0], &req[0]);
                req_valid[0] = true;
            }
            if (col_activate == true && step < comm_per_line-1) {
                MPI_Isend(sub_mat_u, sub_mat_u_row_len * sub_mat_u_col_len, MPI_DOUBLE, u_dest, step+1, col_comm[0], &req[1]);
                req_valid[1] = true;
            }
            if (next_row_activate == true && step < comm_per_line-1) {
                MPI_Irecv(sub_mat_l_buf, sub_mat_l_row_len * sub_mat_l_col_len, MPI_DOUBLE, l_src, step+1, row_comm[0], &req[2]);
                req_valid[2] = true;
            }
            if (next_col_activate == true && step < comm_per_line-1) {
                MPI_Irecv(sub_mat_u_buf, sub_mat_u_row_len * sub_mat_u_col_len, MPI_DOUBLE, u_src, step+1, col_comm[0], &req[3]);
                req_valid[3] = true;
            }
            if (calc_activate == true) {
                matrix_multiply(sub_mat_recon, sub_mat_l, sub_mat_u, sub_mat_row_len, sub_mat_len, sub_mat_col_len, +1);
            }
            for (int i = 0; i < 4; i++) {
                if (req_valid[i] == true) {
                    MPI_Wait(&req[i], MPI_STATUS_IGNORE);
                    req_valid[i] = false;
                }
            }
        }

        /*
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
