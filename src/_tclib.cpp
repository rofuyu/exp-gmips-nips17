#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <omp.h>

#include "fast-pred.h"

extern "C" {
void tc_write(double* W_ptr, double* H_ptr, int32_t m, int32_t n, int32_t k, int32_t topk, const char *filename) {
	omp_set_num_threads(omp_get_max_threads());
    dmat_t W(m, k, W_ptr, ROWMAJOR);
    dmat_t H(n, k, H_ptr, ROWMAJOR);

    test_case_t tc;
    tc.init(W, H.transpose(), topk);
    tc.save_binary_to_file(filename);
}

void tc_read_size(const char *filename, int32_t *m, int32_t *n, int32_t *k) {
    size_t buf;
    FILE *fp = fopen(filename, "rb");
    fseek(fp, sizeof(size_t), SEEK_CUR);
    if(fread(&buf, sizeof(size_t), 1, fp) != 1)
        fprintf(stderr, "Error: wrong input stream in %s.\n", filename);
    *k = static_cast<int32_t>(buf);
    if(fread(&buf, sizeof(size_t), 1, fp) != 1)
        fprintf(stderr, "Error: wrong input stream in %s.\n", filename);
    *n = static_cast<int32_t>(buf);
    fseek(fp, sizeof(double) * (*n) * (*k), SEEK_CUR);
    if(fread(&buf, sizeof(size_t), 1, fp) != 1)
        fprintf(stderr, "Error: wrong input stream in %s.\n", filename);
    *m = static_cast<int32_t>(buf);
    fclose(fp);
}

void tc_read_content(const char *filename, int32_t m, int32_t n, int32_t k, double *W_ptr, double *H_ptr) {
    FILE *fp = fopen(filename, "rb");
    fseek(fp, sizeof(size_t) * 3, SEEK_CUR);
    if(fread(H_ptr, sizeof(double), n * k, fp) != (n * k))
        fprintf(stderr, "Error: wrong input stream in %s abc.\n", filename);
    fseek(fp, sizeof(size_t) * 2, SEEK_CUR);
    if(fread(W_ptr, sizeof(double), m * k, fp) != (m * k))
        fprintf(stderr, "Error: wrong input stream in %s. cde\n", filename);
    fclose(fp);
}

}
