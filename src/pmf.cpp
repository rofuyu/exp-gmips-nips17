#include "pmf.h"

pmf_model_t::pmf_model_t(size_t rows_, size_t cols_, size_t k_, major_t major_type_, bool do_rand_init, val_type global_bias_){ // {{{
	rows = rows_;
	cols = cols_;
	k = k_;
	major_type = major_type_;
	if(do_rand_init)
		rand_init();
	global_bias = global_bias_;
} // }}}

#ifdef _MSC_VER
#define srand48(seed) std::srand(seed)
#define drand48() ((double)std::rand()/(RAND_MAX+1))
#endif

void pmf_model_t::mat_rand_init(dmat_t &X, size_t m, size_t n, long seed) { // {{{
	val_type scale = 1./sqrt(k);
	rng_t rng(seed);
	X = dmat_t(m, n, major_type);
	for(size_t i = 0; i < m; i++)
		for(size_t j = 0; j < n; j++)
			X.at(i,j) = (val_type) rng.uniform((val_type)0.0, scale);
	/*
	//srand48(seed);
	if(major_type == COLMAJOR) {
		X.resize(n, dvec_t(m));
		for(size_t i = 0; i < m; i++)
			for(size_t j = 0; j < n; j++)
				X[j][i] = (val_type) rng.uniform((val_type)0.0, scale);
		//		X[j][i] = (val_type) (scale*(2*drand48()-1.0));
	    //		X[j][i] = (val_type) (scale*(drand48()));
	} else { // major_type == ROWMAJOR
		X.resize(m, dvec_t(n));
		for(size_t i = 0; i < m; i++)
			for(size_t j = 0; j < n; j++)
				X[i][j] = (val_type) rng.uniform((val_type)0.0, scale);
	//			X[i][j] = (val_type) (scale*(2*drand48()-1.0));
	//          X[i][j] = (val_type) (scale*(drand48()));
	}
	*/
} // }}}

void pmf_model_t::rand_init(long seed) { // {{{
	mat_rand_init(W, rows, k, seed);
	mat_rand_init(H, cols, k, seed+k);
} // }}}

val_type pmf_model_t::predict_entry(size_t i, size_t j) const { // {{{
	val_type value = global_bias;
	if(0 <= i && i < rows && 0 <= j && j < cols) {
		for(size_t t = 0; t < k; t++)
			value += W.at(i,t) * H.at(j,t);
	}
	return value;
} // }}}

void pmf_model_t::apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm) { // {{{
	apply_permutation(row_perm.size()==rows? &row_perm[0]: NULL, col_perm.size()==cols? &col_perm[0] : NULL);
} // }}}

void pmf_model_t::apply_permutation(const unsigned *row_perm, const unsigned *col_perm) { // {{{
	W.apply_permutation(row_perm);
	H.apply_permutation(col_perm);
	/*
	if(major_type == COLMAJOR) {
		dvec_t u(rows), v(cols);
		for(size_t t = 0; t < k; t++) {
			dvec_t &Wt = W[t], &Ht = H[t];
			if(row_perm != NULL) {
				for(size_t r = 0; r < rows; r++)
					u[r] = Wt[r];
				for(size_t r = 0; r < rows; r++)
					Wt[r] = u[row_perm[r]];
			}
			if(col_perm != NULL) {
				for(size_t c = 0; c < cols; c++)
					v[c] = Ht[c];
				for(size_t c = 0; c < cols; c++)
					Ht[c] = v[col_perm[c]];
			}
		}
	} else { // major_type == ROWMAJOR
		if(row_perm != NULL) {
			dmat_t buf(rows, dvec_t(k));
			for(size_t r = 0; r < rows; r++)
				for(size_t t = 0; t < k; t++)
					buf[r][t] = W[r][t];

			for(size_t r = 0; r < rows; r++)
				for(size_t t = 0; t < k; t++)
					W[r][t] = buf[row_perm[r]][t];
		}
		if(col_perm != NULL) {
			dmat_t buf(cols, dvec_t(k));
			for(size_t c = 0; c < cols; c++)
				for(size_t t = 0; t < k; t++)
					buf[c][t] = H[c][t];
			for(size_t c = 0; c < cols; c++)
				for(size_t t = 0; t < k; t++)
					H[c][t] = buf[col_perm[c]][t];
		}
	}
	*/
} // }}}

void pmf_model_t::save(FILE *fp) { // {{{
	W.save_binary_to_file(fp);
	H.save_binary_to_file(fp);
	//save_mat_t(W, fp, major_type==ROWMAJOR);
	//save_mat_t(H, fp, major_type==ROWMAJOR);
	double buf = (double)global_bias;
	fwrite(&buf, sizeof(double), 1, fp);
} // }}}

void pmf_model_t::load(FILE *fp, major_t major_type_) { // {{{
	major_type = major_type_;
	W.load_from_binary(fp, major_type_, "pmf-model");
	H.load_from_binary(fp, major_type_, "pmf-model");
	if(major_type_ == ROWMAJOR) {
		W.to_rowmajor();
		H.to_rowmajor();
	} else {
		W.to_colmajor();
		H.to_colmajor();
	}
	//load_mat_t(fp, W, major_type==ROWMAJOR);
	//load_mat_t(fp, H, major_type==ROWMAJOR);
	double buf = 0;
	if(fread(&buf, sizeof(double), 1, fp) != 1)
		fprintf(stderr, "Error: wrong input stream.\n");
	global_bias = (val_type) buf;
	rows = (major_type==ROWMAJOR)? W.size() : W[0].size();
	cols = (major_type==ROWMAJOR)? H.size() : H[0].size();
	k = (major_type==ROWMAJOR)? W[0].size() : W.size();
} // }}}

double pmf_compute_rmse(const smat_t &testY, const dmat_t &W, const dmat_t &H, double global_bias) { // {{{
	double rmse = 0.0;
#pragma omp parallel for schedule(dynamic,50) shared(testY) reduction(+:rmse)
	for(size_t i = 0; i < testY.rows; i++) {
		for(long idx = testY.row_ptr[i]; idx != testY.row_ptr[i+1]; idx++) {
			size_t j = testY.col_idx[idx];
			double true_val = testY.val_t[idx], pred_val = 0.0;
			for(size_t t = 0; t < W.cols; t++)
				pred_val += W.at(i,t)*H.at(j,t);
			rmse += (pred_val-true_val)*(pred_val-true_val);
		}
	}
	return sqrt(rmse/(double)testY.nnz);
} // }}}

// Save a dmat_t A to a file in row_major order.
// row_major = true: A is stored in row_major order,
// row_major = false: A is stored in col_major order.
void save_mat_t(const dmat_t &A, FILE *fp, bool row_major){//{{{
	if (fp == NULL)
		fprintf(stderr, "output stream is not valid.\n");
	long m = row_major? A.size(): A[0].size();
	long n = row_major? A[0].size(): A.size();
	fwrite(&m, sizeof(long), 1, fp);
	fwrite(&n, sizeof(long), 1, fp);
	double *buf = MALLOC(double, m*n);

	if (row_major) {
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
				buf[idx++] = A[i][j];
	} else {
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
				buf[idx++] = A[j][i];
	}
	fwrite(&buf[0], sizeof(double), m*n, fp);
	fflush(fp);
	free(buf);
}//}}}

// Load a matrix from a file and return a dmat_t matrix
// row_major = true: the returned A is stored in row_major order,
// row_major = false: the returened A  is stored in col_major order.
void load_mat_t(FILE *fp, dmat_t &A, bool row_major){//{{{
	if (fp == NULL)
		fprintf(stderr, "Error: null input stream.\n");
	long m, n;
	if(fread(&m, sizeof(long), 1, fp) != 1)
		fprintf(stderr, "Error: wrong input stream.\n");
	if(fread(&n, sizeof(long), 1, fp) != 1)
		fprintf(stderr, "Error: wrong input stream.\n");
	double *buf = MALLOC(double, m*n);
	if(fread(&buf[0], sizeof(val_type), m*n, fp) != m*n)
		fprintf(stderr, "Error: wrong input stream.\n");
	if (row_major) {
		//A = dmat_t(m, dvec_t(n));
		A.resize(m, dvec_t(n));
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
				A[i][j] = buf[idx++];
	} else {
		//A = dmat_t(n, dvec_t(m));
		A.resize(n, dvec_t(m));
		size_t idx = 0;
		for(size_t i = 0; i < m; ++i)
			for(size_t j = 0; j < n; ++j)
				A[j][i] = buf[idx++];
	}
	free(buf);
}//}}}

// load utility for CCS RCS
void pmf_read_data(const char* srcdir, smat_t &training_set, smat_t &test_set, smat_t::format_t fmt) { //{{{
	size_t m, n, nnz;
	char filename[1024], buf[1024], suffix[12];
	FILE *fp;
	sprintf(filename,"%s/meta",srcdir);
	fp = fopen(filename,"r");
	if(fscanf(fp, "%lu %lu", &m, &n) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 1 of %s\n", srcdir);
		return;
	}

	if(fscanf(fp, "%lu %s", &nnz, buf) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 2 of %s\n", srcdir);
		return;
	}
	if(fmt == smat_t::TXT)
		suffix[0] = 0; //sprintf(suffix, "");
	else if(fmt == smat_t::PETSc)
		sprintf(suffix, ".petsc");
	else
		printf("Error: fmt %d is not supported.", fmt);

	sprintf(filename,"%s/%s%s", srcdir, buf, suffix);
	training_set.load(m, n, nnz, filename, fmt);

	if(fscanf(fp, "%lu %s", &nnz, buf) != EOF){
		sprintf(filename,"%s/%s%s", srcdir, buf, suffix);
		test_set.load(m, n, nnz, filename, fmt);
	}
	fclose(fp);
	return ;
}//}}}

// load utility for blocks_t
void pmf_read_data(const char* srcdir, blocks_t &training_set, blocks_t &test_set, smat_t::format_t fmt) { //{{{
	size_t m, n, nnz;
	char filename[1024], buf[1024], suffix[12];
	FILE *fp;
	sprintf(filename,"%s/meta",srcdir);
	fp = fopen(filename,"r");
	if(fscanf(fp, "%lu %lu", &m, &n) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 1 of %s\n", srcdir);
		return;
	}

	if(fscanf(fp, "%lu %s", &nnz, buf) != 2) {
		fprintf(stderr, "Error: corrupted meta in line 2 of %s\n", srcdir);
		return;
	}
	suffix[0] = 0; // TXT
	if(fmt == smat_t::TXT)
		suffix[0] = 0; //sprintf(suffix, "");
	else if(fmt == smat_t::PETSc)
		sprintf(suffix, ".petsc");
	else
		printf("Error: fmt %d is not supported.", fmt);

	sprintf(filename,"%s/%s%s", srcdir, buf, suffix);
	training_set.load(m, n, nnz, filename, fmt);
	//training_set.load(m, n, nnz, filename);

	if(fscanf(fp, "%lu %s", &nnz, buf) != EOF){
		sprintf(filename,"%s/%s%s", srcdir, buf, suffix);
		test_set.load(m, n, nnz, filename, fmt);
		//test_set.load(m, n, nnz, filename);
	}
	fclose(fp);
	return ;
}//}}}

