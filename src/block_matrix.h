#ifndef BLOCK_MATRIX_H
#define BLOCK_MATRIX_H

#include "sparse_matrix.h"

#define blocks_t block_matrix
template<typename val_type> class blocks_t;

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

// a block of rate set for PSGD
template<typename val_type>
class blocks_t {
	public:
		typedef entry_t<val_type> rate_t;
		typedef dense_vector<rate_t> entry_set_t;
		struct block_t { // {{{
			size_t start_row, sub_rows;
			size_t start_col, sub_cols;
			entry_set_t entry_set;
			blocks_t *blocks;
			block_t() {}
			block_t(size_t start_row, size_t sub_rows, size_t start_col, size_t sub_cols, const entry_set_t& entry_set, blocks_t *blocks=NULL):
				start_row(start_row), sub_rows(sub_rows), start_col(start_col), sub_cols(sub_cols), entry_set(entry_set), blocks(blocks){}
			rate_t & operator[](size_t idx) { return entry_set[idx];}
			const rate_t & operator[](size_t idx) const { return entry_set[idx];}
			size_t size() const {return entry_set.size();}
			const rate_t* find(const rate_t &val) const {
				const rate_t *first = entry_set.data(), *last = first+entry_set.size();;
				first = std::lower_bound(first, last, val, RateCompare(blocks));
				if(first != last && val.i == first->i && val.j == first->j)
					return first;
				else
					return NULL;
			}
		}; // }}}

		size_t rows, cols, nnz;
		int B, Bm, Bn;
		std::vector<rate_t> allrates;
		std::vector<unsigned> nnz_row, nnz_col;
		bool rowmajored;

		// Constructors
		blocks_t():rows(0),cols(0),nnz(0),B(0),Bm(0),Bn(0){}


		// Access methods
		void set_blocks(int B_, bool rowmajor=true){ // {{{
			B = B_;
			Bm = rows/B+((rows%B)?1:0); // block's row size
			Bn = cols/B+((cols%B)?1:0); // block's col size
			rowmajored = rowmajor;
			block_ptr.clear(); block_ptr.resize(B*B+1,0);
			nnz_row.clear(); nnz_row.resize(rows);
			nnz_col.clear(); nnz_col.resize(cols);
			for(size_t idx = 0; idx < allrates.size(); idx++) {
				const rate_t &r = allrates[idx];
				block_ptr[bid_of_rate(r.i, r.j)+1]++;
				nnz_row[r.i]++;
				nnz_col[r.j]++;
			}
			for(int bid = 1; bid <= B*B; ++bid)
				block_ptr[bid] += block_ptr[bid-1];
			std::sort(allrates.begin(), allrates.end(), RateCompare(this));
			block_set.resize(B*B);
			for(int bid = 0; bid < B*B; bid++) {
				size_t start_row = (bid/B)*Bm, sub_rows = (start_row+Bm)<=rows ? Bm : (start_row+Bm-rows);
				size_t start_col = (bid%B)*Bn, sub_cols = (start_col+Bn)<=cols ? Bn : (start_col+Bn-cols);
				block_set[bid] = block_t(start_row, sub_rows, start_col, sub_cols,
						entry_set_t(block_ptr[bid+1]-block_ptr[bid],&allrates[block_ptr[bid]]), this);
			}

		} // }}}
		int bid_of_rate(int i, int j) const {return (i/Bm)*B + (j/Bn);}
		int get_bid(int bi, int bj) const {return bi*B+bj;}
		int size() const {return B*B;}
		block_t& operator[] (int bid) { return block_set[bid]; }
		const block_t& operator[] (int bid) const { return block_set[bid]; }
		void apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm) { // {{{
			apply_permutation(row_perm.size()==rows? &row_perm[0]: NULL, col_perm.size()==cols? &col_perm[0]: NULL);
		} // }}}
		void apply_permutation(const unsigned *row_perm=NULL, const unsigned *col_perm=NULL) { // {{{
			if(row_perm)
				for(long idx = 0; idx < (long)nnz; idx++) {
					rate_t &r = allrates[idx];
					r.i = row_perm[r.i];
				}
			if(col_perm)
				for(long idx = 0; idx < (long)nnz; idx++) {
					rate_t &r = allrates[idx];
					r.j = col_perm[r.j];
				}
		} // }}}

		size_t nnz_of_row(unsigned i) const {return nnz_row[i];}
		size_t nnz_of_col(unsigned i) const {return nnz_col[i];}

		// IO methods
		void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t<val_type>* entry_it){  // {{{
			rows =_rows,cols=_cols,nnz=_nnz;
			allrates.reserve(nnz);
			nnz_row.clear(); nnz_row.resize(rows);
			nnz_col.clear(); nnz_col.resize(cols);
			for(size_t idx=0; idx < nnz; ++idx) {
				rate_t r = entry_it->next();
				allrates.push_back(r);
				nnz_row[r.i]++;
				nnz_col[r.j]++;
			}

		} // }}}
		void load(long _rows, long _cols, long _nnz, const char *filename, typename sparse_matrix<val_type>::format_t fmt){  // {{{
			if(fmt == sparse_matrix<val_type>::TXT) {
				file_iterator_t<val_type> entry_it(_nnz, filename);
				load_from_iterator(_rows, _cols, _nnz, &entry_it);
			} else if(fmt == sparse_matrix<val_type>::PETSc) {
				load_from_PETSc(filename);
			} else {
				fprintf(stderr, "Error: filetype %d not supported\n", fmt);
				return ;
			}
		} // }}}
		void load_from_PETSc(const char *filename);

		// used for MPI verions
		void from_mpi(long _rows, long _cols, long _nnz){ // {{{
			rows =_rows,cols=_cols,nnz=_nnz;
			block_ptr.resize(B+1);
			allrates.resize(nnz);
			nnz_row.resize(rows);
			nnz_col.resize(cols);
			Bm = rows/B+((rows%B)?1:0); // block's row size
			Bn = cols/B+((cols%B)?1:0); // block's col size
		} // }}}

		val_type get_global_mean() const { // {{{
			val_type sum=0;
			for(int i=0; i < nnz; ++i)
				sum+=allrates[i].v;
			return sum/(val_type)nnz;
		} // }}}
		void remove_bias(val_type bias=0){ // {{{
			if(bias)
				for(int i=0; i < nnz; ++i) allrates[i].v -= bias;
		} // }}}

	private:
		struct RateCompare { // {{{
			bool rowmajor;
			blocks_t *ptr;
			RateCompare(blocks_t *ptr):ptr(ptr), rowmajor(ptr->rowmajored){}
			bool operator()(const rate_t &x, const rate_t &y) const {
				int x_bid = ptr->bid_of_rate(x.i, x.j), y_bid = ptr->bid_of_rate(y.i, y.j);
				if(x_bid != y_bid)
					return x_bid < y_bid;
				if(rowmajor) {
					if(x.i != y.i)
						return x.i < y.i;
					else
						return x.j < y.j;
				} else {
					if(x.j != y.j)
						return x.j < y.j;
					else
						return x.i < y.i;
				}
			}
		}; // }}}
		std::vector<block_t> block_set;
		std::vector<long> block_ptr;
};


template<typename val_type>
void blocks_t<val_type>::load_from_PETSc(const char *filename) { // {{{
	const int UNSIGNED_FILE = 1211216, LONG_FILE = 1015;
	int32_t int_buf[3];
	size_t headersize = 0;
	FILE *fp = fopen(filename, "rb");
	if(fp == NULL) {
		fprintf(stderr, "Error: can't read the file (%s)!!\n", filename);
		return;
	}
	headersize += sizeof(int)*fread(int_buf, sizeof(int), 3, fp);
	int filetype = int_buf[0];
	rows = (size_t) int_buf[1];
	cols = (size_t) int_buf[2];
	if(filetype == UNSIGNED_FILE) {
		headersize += sizeof(int)*fread(int_buf, sizeof(int32_t), 1, fp);
		nnz = (size_t) int_buf[0];
	} else if (filetype == LONG_FILE){
		headersize += sizeof(size_t)*fread(&nnz, sizeof(int64_t), 1, fp);
	} else {
		fprintf(stderr, "Error: wrong PETSc format for %s\n", filename);
	}
	// Allocation of memory
	allrates.resize(nnz);
	nnz_row.clear(); nnz_row.resize(rows);
	nnz_col.clear(); nnz_col.resize(cols);


	// load blocks_t from the binary PETSc format
	{
		{ // read row_idx {{{
			std::vector<int32_t> nnz_row(rows);
			headersize += sizeof(int32_t)*fread(&nnz_row[0], sizeof(int32_t), rows, fp);
			for(size_t r = 0, idx = 0; r < rows; r++)
				for(size_t t = 0; t < nnz_row[r]; t++) {
					allrates[idx++].i = (unsigned) r;
					nnz_row[r]++;
				}
		} // }}}

		{ // read col_idx {{{
			const size_t chunksize = 1024;
			unsigned buf[chunksize];
			size_t idx = 0;
			while(idx + chunksize < nnz) {
				headersize += sizeof(unsigned)*fread(&buf[0], sizeof(unsigned), chunksize, fp);
				for(size_t i = 0; i < chunksize; i++) {
					allrates[idx+i].j = (unsigned) buf[i];
					nnz_col[buf[i]]++;
				}
				idx += chunksize;
			}
			size_t remaining = nnz - idx;
			headersize += sizeof(unsigned)*fread(&buf[0], sizeof(unsigned), remaining, fp);
			for(size_t i = 0; i < remaining; i++) {
				allrates[idx+i].j = (unsigned) buf[i];
				nnz_col[buf[i]]++;
			}
		} // }}}

		{ // read val_t {{{
			const size_t chunksize = 1024;
			double buf[chunksize];
			size_t idx = 0;
			while(idx + chunksize < nnz) {
				headersize += sizeof(double)*fread(&buf[0], sizeof(double), chunksize, fp);
				for(size_t i = 0; i < chunksize; i++)
					allrates[idx+i].j = (double) buf[i];
				idx += chunksize;
			}
			size_t remaining = nnz - idx;
			headersize += sizeof(double)*fread(&buf[0], sizeof(double), remaining, fp);
			for(size_t i = 0; i < remaining; i++)
				allrates[idx+i].j = (double) buf[i];
		} // }}}
	}
	fclose(fp);
} // }}}


// blocks_t iterator
/*
template<typename val_type>
class blocks_iterator_t: public entry_iterator_t<val_type>{
	public:
		enum {ROWMAJOR, COLMAJOR};
		blocks_iterator_t(const blocks_t<val_type>& M, int major = ROWMAJOR);
		~blocks_iterator_t() {}
		entry_t<val_type> next();
	private:
		size_t nnz;
		unsigned *col_idx;
		size_t *row_ptr;
		val_type *val_t;
		size_t rows, cols, cur_idx;
		size_t cur_row;
};
*/

#undef blocks_t

#endif // BLOCK_MATRIX_H
