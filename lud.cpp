#include <cstdio>
#include <cstdlib>
#include <cmath>

#define min(x, y) ((x) < (y) ? (x) : (y))

class Matrix2D {
public:
    const int len;
    const int row_len;
    const int col_len;
    const int size;
    double *data;

    Matrix2D(int _row_len, int _col_len)
        : row_len(_row_len), col_len(_col_len), len(_col_len), size(_row_len * _col_len) {
        data = new double[_row_len * _col_len];
    }
    ~Matrix2D() {
        delete [] data;
    }
    inline double& operator()(int row, int col) {
        return data[row * col_len + col];
    }
    Matrix2D& clear() {
        for (int i = 0; i < row_len; i++) {
            for (int j = 0; j < col_len; j++) {
                data[i * col_len + j] = 0;
            }
        }
        return *this;
    }
    void lu_decompose_to(Matrix2D& mat_l, Matrix2D& mat_u) {
        for (int k = 0; k < len; k++) {
            mat_l(k, k) = 1;
            mat_u(k, k) = data[k * col_len + k];
            for (int i = k+1; i < row_len; i++) {
                mat_l(i, k) = (long double)data[i * col_len + k] / (long double)mat_u(k, k);
                mat_u(k, i) = data[k * col_len + i];
            }
            for (int i = k+1; i < row_len; i++) {
                for (int j = k+1; j < col_len; j++) {
                    data[i * col_len + j] -= (long double)mat_l(i, k) * (long double)mat_u(k, j);
                }
            }
        }
    }
};

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <seed_number>\n", argv[0]);
        return -1;
    }

    int mat_len = atoi(argv[1]);
    int seed = atoi(argv[2]);

    Matrix2D mat(mat_len, mat_len);
    Matrix2D mat_stage(mat_len, mat_len);
    Matrix2D mat_l(mat_len, mat_len);
    Matrix2D mat_u(mat_len, mat_len);

    // Build Matrix
    srand(seed);
    for (int i = 0; i < mat_len; i++) {
        for (int j = 0; j < mat_len; j++) {
            mat(i, j) = mat_stage(i, j) = (int)rand() / 1000;
        }
    }

    // LU Decomposition
    mat_stage.lu_decompose_to(mat_l.clear(), mat_u.clear());

    // Reconstruct Matrix
    Matrix2D& mat_recon = mat_stage.clear();
    for (int i = 0; i < mat_len; i++) {
        for (int j = 0; j < mat_len; j++) {
            int min_idx = min(i, j);
            for (int k = 0; k <= min_idx; k++) {
                mat_stage(i, j) += mat_l(i, k) * mat_u(k, j);
            }
        }
    }

    // Calculate Matrix Difference
    double diff = 0;
    for (int j = 0; j < mat_recon.col_len; j++) {
        double row_sum = 0;
        for (int i = 0; i < mat_recon.row_len; i++) {
            row_sum += (mat(i, j) - mat_recon(i, j)) * (mat(i, j) - mat_recon(i, j));
        }
        diff += sqrt(row_sum);
    }
    printf("%lf\n", diff);

    return 0;
}
