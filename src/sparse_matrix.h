#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

// headers {{{
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstddef>
#include <assert.h>
#include <omp.h>


#include <iostream>

#ifdef _MSC_VER
#if _MSC_VER >= 1600
#include <cstdint>
#else
typedef __int8              int8_t;
typedef __int16             int16_t;
typedef __int32             int32_t;
typedef __int64             int64_t;
typedef unsigned __int8     uint8_t;
typedef unsigned __int16    uint16_t;
typedef unsigned __int32    uint32_t;
typedef unsigned __int64    uint64_t;
#endif
#endif

#if __cplusplus >= 201103L || (defined(_MSC_VER) && (_MSC_VER >= 1500)) // Visual Studio 2008
#define CPP11
#endif

/* random number genrator: simulate the interface of python random module*/
#include <limits>
#if defined(CPP11)
#include <random>
template<typename engine_t=std::mt19937>
struct random_number_generator : public engine_t { // {{{
	typedef typename engine_t::result_type result_type;

	random_number_generator(unsigned seed=0): engine_t(seed){ }

	result_type randrange(result_type end=engine_t::max()) { return engine_t::operator()() % end; }
	template<class T=double, class T2=double> T uniform(T start=0.0, T2 end=1.0) {
		return std::uniform_real_distribution<T>(start, (T)end)(*this);
	}
	template<class T=double> T normal(T mean=0.0, T stddev=1.0) {
		return std::normal_distribution<T>(mean, stddev)(*this);
	}
	template<class T=int, class T2=T> T randint(T start=0, T2 end=std::numeric_limits<T>::max()) {
		return std::uniform_int_distribution<T>(start, end)(*this);
	}
	template<class RandIter> void shuffle(RandIter first, RandIter last) {
		std::shuffle(first, last, *this);
	}
};
#else
#include <tr1/random>
template<typename engine_t=std::tr1::mt19937>
struct random_number_generator : public engine_t {
	typedef typename engine_t::result_type result_type;

	random_number_generator(unsigned seed=0): engine_t(seed) { }
	result_type operator()() { return engine_t::operator()(); }
	result_type operator()(result_type n) { return randint(result_type(0), result_type(n-1)); }

	result_type randrange(result_type end=engine_t::max()) { return engine_t::operator()() % end; }
	template<class T, class T2> T uniform(T start=0.0, T2 end=1.0) {
		typedef std::tr1::uniform_real<T> dist_t;
		return std::tr1::variate_generator<engine_t*, dist_t>(this, dist_t(start,(T)end))();
	}
	template<class T, class T2> T normal(T mean=0.0, T2 stddev=1.0) {
		typedef std::tr1::normal_distribution<T> dist_t;
		return std::tr1::variate_generator<engine_t*, dist_t>(this, dist_t(mean, (T)stddev))();
	}
	template<class T, class T2> T randint(T start=0, T2 end=std::numeric_limits<T>::max()) {
		typedef std::tr1::uniform_int<T> dist_t;
		return std::tr1::variate_generator<engine_t*, dist_t>(this, dist_t(start,end))();
	}
	template<class RandIter> void shuffle(RandIter first, RandIter last) {
		std::random_shuffle(first, last, *this);
	}
}; // }}}
#endif
typedef random_number_generator<> rng_t;

template<typename T>
void gen_permutation_pair(size_t size, std::vector<T> &perm, std::vector<T> &inv_perm, int seed=0) { // {{{
	perm.resize(size);
	for(size_t i = 0; i < size; i++)
		perm[i] = i;

	rng_t rng(seed);
	rng.shuffle(perm.begin(), perm.end());
	//std::srand(seed);
	//std::random_shuffle(perm.begin(), perm.end());

	inv_perm.resize(size);
	for(size_t i = 0; i < size; i++)
		inv_perm[perm[i]] = i;
} // }}}

//#include "zlib_util.h"
// }}}

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))
#define CALLOC(type, size) (type*)calloc((size), sizeof(type))
#define REALLOC(ptr, type, size) (type*)realloc((ptr), sizeof(type)*(size))

//namespace rofu {
typedef unsigned major_t;
const major_t ROWMAJOR = 0U;
const major_t COLMAJOR = 1U;
const major_t default_major = COLMAJOR;

// Zip Iterator
// Commom usage: std::sort(zip_iter(A.begin(),B.begin()), zip_iter(A.end(),B.end()));
template<class T1, class T2> struct zip_body;
template<class T1, class T2> struct zip_ref;
template<class IterT1, class IterT2> struct zip_it;
template<class IterT1, class IterT2> zip_it<IterT1, IterT2> zip_iter(IterT1 x, IterT2 y);

#define dvec_t dense_vector
template<typename val_type> class dvec_t;
#define dmat_t dense_matrix
template<typename val_type> class dmat_t;
#define smat_t sparse_matrix
template<typename val_type> class smat_t;
#define eye_t identity_matrix
template<typename val_type> class eye_t;
#define gmat_t general_matrix
template<typename val_type> class gmat_t { // {{{
	public:
		size_t rows, cols;
		gmat_t(size_t rows=0, size_t cols=0): rows(rows), cols(cols){}
		virtual bool is_sparse() const {return false;}
		virtual bool is_dense() const {return false;}
		virtual bool is_identity() const {return false;}
		smat_t<val_type>& get_sparse() {assert(is_sparse()); return static_cast<smat_t<val_type>&>(*this);}
		const smat_t<val_type>& get_sparse() const {assert(is_sparse()); return static_cast<const smat_t<val_type>&>(*this);}
		dmat_t<val_type>& get_dense() {assert(is_dense()); return static_cast<dmat_t<val_type>&>(*this);}
		const dmat_t<val_type>& get_dense() const {assert(is_dense()); return static_cast<const dmat_t<val_type>&>(*this);}
}; // }}}

template<typename val_type> class entry_iterator_t; // iterator for files with (i,j,v) tuples
template<typename val_type> class smat_iterator_t; // iterator for nonzero entries in smat_t
template<typename val_type> class smat_subset_iterator_t; // iterator for nonzero entries in a subset
template<typename val_type> class dmat_iterator_t; // iterator for nonzero entries in dmat_t

// H = X*W, (X: m*n, W: n*k row-major, H m*k row major)
template<typename val_type> void smat_x_dmat(const smat_t<val_type> &X, const val_type* W, const size_t k, val_type *H);
template<typename val_type> void smat_x_dmat(const smat_t<val_type> &X, const dmat_t<val_type> &W, dmat_t<val_type> &H);
template<typename val_type> void gmat_x_dmat(const gmat_t<val_type> &X, const dmat_t<val_type> &W, dmat_t<val_type> &H);

// H = a*X*W + H0, (X: m*n, W: n*k row-major, H m*k row major)
template<typename val_type, typename T2> void smat_x_dmat(T2 a, const smat_t<val_type> &X, const val_type* W, const size_t k, const val_type *H0, val_type *H);
template<typename val_type, typename T2> void smat_x_dmat(T2 a, const smat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H);


// H = a*X*W + b*H0, (X: m*n, W: n*k row-major, H m*k row major)
template<typename val_type, typename T2, typename T3>
void smat_x_dmat(T2 a, const smat_t<val_type>& X, const val_type *W, const size_t k, T3 b, const val_type *H0, val_type *H);
template<typename val_type, typename T2, typename T3>
void smat_x_dmat(T2 a, const smat_t<val_type> &X, const dmat_t<val_type> &W, T3 b, const dmat_t<val_type> &H0, dmat_t<val_type> &H);
template<typename val_type, typename T2, typename T3>
void gmat_x_dmat(T2 a, const gmat_t<val_type> &X, const dmat_t<val_type> &W, T3 b, const dmat_t<val_type> &H0, dmat_t<val_type> &H);

// trace(W'*X*H)
template<typename val_type> val_type trace_dmat_T_smat_dmat(const dmat_t<val_type> &W, const smat_t<val_type> &X, const dmat_t<val_type> &H);

// Dense Vector
template<typename val_type>
class dvec_t { // {{{
	friend class dmat_t<val_type>;
	private:
		bool mem_alloc_by_me;
		void zero_init() {len = 0; buf = NULL; mem_alloc_by_me = false;}
	public:
		size_t len;
		val_type *buf;

		// Default Constructor
		dvec_t() {zero_init();}
		// Copy Constructor
		dvec_t(const dvec_t& v) { // {{{
			zero_init();
			*this = v;
		} // }}}
		// Copy Assignment
		dvec_t& operator=(const dvec_t& other) { // {{{
			if(this == &other) return *this;
			if(other.is_view()) {  // view to view copy
				if(mem_alloc_by_me) clear_space();
				memcpy(this, &other, sizeof(dvec_t));
			} else { // deep to deep copy
				resize(other.size());
				memcpy(buf, other.buf, sizeof(val_type)*len);
			}
			return *this;
		} // }}}
		// View Constructor: allocate space if buf == NULL
		explicit dvec_t(size_t len, val_type *buf=NULL): len(len), buf(buf), mem_alloc_by_me(false) { // {{{
			if(buf == NULL && len != 0) {
				this->buf = MALLOC(val_type, len);
				memset(this->buf, 0, sizeof(val_type)*len);
				mem_alloc_by_me = true;
			}
		} // }}}
		// Fill Constructor
		explicit dvec_t(size_t len, const val_type &x) {zero_init();resize(len,x);}
		// dense_matrix_t Converter
		dvec_t(const dmat_t<val_type>& m) { // {{{
			//puts("dvect dmat convert ctor");
			zero_init();
			if(m.is_view()) {len=m.rows*m.cols; buf=m.buf;}
			else {
				resize(m.rows*m.cols);
				memcpy(buf, m.buf, sizeof(val_type)*len);
			}
		} // }}}

#if defined(CPP11)
		// Move Constructor
		dvec_t(dvec_t&& m){
			zero_init(); *this = std::move(m);}
		// Move Assignment
		dvec_t& operator=(dvec_t&& other) { // {{{
			if(this == &other) return *this;
			clear_space();
			memcpy(this, &other, sizeof(dvec_t));
			other.zero_init();
			return *this;
		} // }}}
#endif
		~dvec_t() {clear_space(); }

		bool is_view() const {return mem_alloc_by_me==false;}
		void clear_space() {if(mem_alloc_by_me) free(buf); zero_init();}
		dvec_t get_view() const {return dvec_t(len, buf);}
		dvec_t& grow_body() { // {{{
			if(is_view()) {
				dvec_t tmp_view = *this;
				this->resize(len);
				memcpy(buf, tmp_view.buf, sizeof(val_type)*len);
			}
			return *this;
		} // }}}
		dvec_t& assign(const dvec_t& other) { // {{{
			assert(len == other.len);
			return assign((val_type)1.0, other);
		} // }}}
		template<typename T>
		dvec_t& assign(T a, const dvec_t& other) { // {{{
			assert(len == other.len);
			if(a == T(0))
				memset(buf, 0, sizeof(val_type)*len);
			else if(a == T(1)) {
				if(this == &other)
					return *this;
#pragma omp parallel for schedule(static)
				for(size_t idx = 0; idx < len; idx++)
					at(idx) = other.at(idx);
			} else {
#pragma omp parallel for schedule(static)
				for(size_t idx = 0; idx < len; idx++)
					at(idx) = a*other.at(idx);
			}
			return *this;
		} // }}}

		size_t size() const {return len;};
		void resize(size_t len_, const val_type &x) { // {{{
			resize(len_);
			for(size_t i = 0; i < len; i++)
				buf[i] = x;
		} // }}}
		void resize(size_t len_) { // {{{
			if(mem_alloc_by_me)
				buf = REALLOC(buf, val_type, len_);
			else
				buf = MALLOC(val_type, len_);
			mem_alloc_by_me = true;
			len = len_;
		} // }}}
		val_type& at(size_t idx) {return buf[idx];}
		const val_type& at(size_t idx) const {return buf[idx];}
		val_type& operator[](size_t idx) {return buf[idx];}
		const val_type& operator[](size_t idx) const {return buf[idx];}
		val_type* data() {return buf;}
		const val_type* data() const {return buf;}
		void print(const char *str="") const {
			printf("%s dvec_t: len %d, is_view %d, buf %p\n", str, len, is_view(), buf);
			for(size_t i = 0; i < len; i ++)
				printf("%g ", buf[i]);
			puts("");
		}
}; // }}}

// Dense Matrix
template<typename val_type>
class dmat_t : public gmat_t<val_type> { // {{{
	friend class dvec_t<val_type>;
	public:
		// size_t rows, cols; inherited from gmat_t
		using gmat_t<val_type>::rows;
		using gmat_t<val_type>::cols;
		val_type *buf;

		static dmat_t rand(rng_t &rng, size_t m, size_t n, double lower=0.0, double upper=1.0, major_t major_type_=default_major) { // {{{
			dmat_t ret(m, n, major_type_);
			if(lower >= upper) lower = upper;
			for(size_t idx = 0; idx < m*n; idx++)
				ret.buf[idx] = (val_type)rng.uniform(lower, upper);
			return ret;
		} // }}}
		static dmat_t randn(rng_t &rng, size_t m, size_t n, double mean=0.0, double std=1.0, major_t major_type_=default_major) { // {{{
			dmat_t ret(m, n, major_type_);
			for(size_t idx = 0; idx < m*n; idx++)
				ret.buf[idx] = (val_type)rng.normal(mean, std);
			return ret;
		} // }}}
	private:
		bool mem_alloc_by_me;
		major_t major_type;
		typedef dvec_t<val_type> vec_t;
		std::vector<vec_t> vec_set; // view for each row/col depending on the major_type;
		void zero_init() {rows=cols=0; buf=NULL; major_type=default_major; mem_alloc_by_me=false; vec_set.clear();}
		void init_vec_set() { // {{{
			if(is_rowmajor()) {
				vec_set.resize(rows);
				for(size_t r = 0; r < rows; r++)
					vec_set[r] = dvec_t<val_type>(cols, &buf[r*cols]);
			} else {
				vec_set.resize(cols);
				for(size_t c = 0; c < cols; c++)
					vec_set[c] = dvec_t<val_type>(rows, &buf[c*rows]);
			}
		} // }}}
		void inv_major() { // {{{
			if(rows == 1 || cols == 1) {
				major_type = is_rowmajor()? COLMAJOR: ROWMAJOR;
				init_vec_set();
			} else if(rows == cols && !is_view()) { // inplace for square matrix
				for(size_t r = 0; r < rows; r++)
					for(size_t c = 0; c < r; c++)
						std::swap(at(r,c),at(c,r));
				major_type = is_rowmajor()? COLMAJOR: ROWMAJOR;
			} else {
				dmat_t tmp(*this);
				major_type = is_rowmajor()? COLMAJOR: ROWMAJOR;
				resize(rows,cols);
				for(size_t r = 0; r < rows; r++)
					for(size_t c = 0; c < cols; c++)
						at(r,c) = tmp.at(r,c);
			}
		} // }}}
	public:
		// Default Constructor
		dmat_t() {zero_init();}
		// Copy Constructor
		dmat_t(const dmat_t& other, major_t major_type_=default_major) { // {{{
			zero_init();
			if(other.major_type == major_type_)
				*this = other;
			else { // deep copy is required when major_type changes
				major_type = major_type_;
				resize(other.rows, other.cols);
				for(size_t r = 0; r < rows; r++)
					for(size_t c = 0; c < cols; c++)
						at(r,c) = other.at(r,c);
			}
		} // }}}
		// Copy Assignment
		dmat_t& operator=(const dmat_t& other) { // {{{
			if(this == &other) return *this;
			if(other.is_view()) { // for view
				if(mem_alloc_by_me) clear_space();
				rows = other.rows; cols = other.cols; buf = other.buf; major_type = other.major_type;
				init_vec_set();
				mem_alloc_by_me = false;
			} else { // deep copy
				if(is_view() || rows!=other.rows || cols!=other.cols || major_type!=other.major_type) {
					major_type = other.major_type;
					resize(other.rows, other.cols);
				}
				memcpy(buf, other.buf, sizeof(val_type)*rows*cols);
			}
			return *this;
		} // }}}
		// View Constructor: allocate space if buf_ == NULL
		explicit dmat_t(size_t rows_, size_t cols_, major_t major_type=default_major): gmat_t<val_type>(rows_,cols_), buf(NULL), mem_alloc_by_me(false),  major_type(major_type) { // {{{
			resize(rows,cols);
			memset(this->buf, 0, sizeof(val_type)*rows*cols);
		} // }}}
		explicit dmat_t(size_t rows_, size_t cols_, val_type *buf, major_t major_type_): gmat_t<val_type>(rows_,cols_), buf(buf), mem_alloc_by_me(false), major_type(major_type_) { // {{{
			init_vec_set();
		} // }}}
		// Fill Constructor
		explicit dmat_t(size_t nr_copy, const dvec_t<val_type>& v, major_t major_type_=default_major) { // {{{
			zero_init();
			major_type = major_type_;
			resize(nr_copy, v);
		} // }}}
		// dense_vector Converter
		dmat_t(const dvec_t<val_type>& v, major_t major_type_=default_major) { // {{{
			zero_init();
			major_type = major_type_;
			if(!v.is_view())
				resize(1, v);
			else {
				rows = is_rowmajor()? 1: v.size();
				cols = is_colmajor()? 1: v.size();
				buf = v.buf;
				init_vec_set();
			}
		} // }}}
		template<typename T>
		dmat_t(const smat_t<T>& sm, major_t major_type_=default_major) { // {{{
			zero_init();
			major_type = major_type_;
			resize(sm.rows, sm.cols);
			memset(buf, 0, sizeof(val_type)*rows*cols);
			for(size_t i = 0; i < sm.rows; i++)
				for(size_t idx = sm.row_ptr[i]; idx != sm.row_ptr[i+1]; idx++)
					at(i, sm.col_idx[idx]) = sm.val_t[idx];
		} // }}}
		template<typename T>
		dmat_t(const eye_t<T>& eye, major_t major_type_=default_major) { // {{{
			zero_init();
			major_type = major_type_;
			resize(eye.rows, eye.cols);
			memset(buf, 0, sizeof(val_type)*rows*cols);
			for(size_t i = 0; i < rows; i++)
					at(i,i) = 1;
		} // }}}

#if defined(CPP11)
		// Move Constructor
		dmat_t(dmat_t&& m){
			zero_init();
			*this = std::move(m);
		}
		// Move Assignment
		dmat_t& operator=(dmat_t&& other) { // {{{
			if(this == &other) return *this;
			clear_space();
			rows = other.rows;
			cols = other.cols;
			buf = other.buf;
			vec_set = std::move(other.vec_set);
			mem_alloc_by_me = other.mem_alloc_by_me;
			major_type = other.major_type;
			other.zero_init();
			return *this;
		} // }}}
#endif
		~dmat_t() {if(mem_alloc_by_me) {for(size_t i = 0; i < rows*cols; i++) buf[i]=-1;}clear_space();}


		bool is_view() const {return mem_alloc_by_me==false;}
		bool is_dense() const {return true;}
		bool is_rowmajor() const {return major_type==ROWMAJOR;}
		bool is_colmajor() const {return major_type==COLMAJOR;}

		void clear_space() {if(mem_alloc_by_me) free(buf); zero_init();}
		dmat_t get_view() const {return dmat_t(rows,cols,buf,major_type);}
		dmat_t& grow_body() { // {{{
			if(is_view()) {
				dmat_t tmp_view = *this;
				this->resize(rows,cols);
				memcpy(buf, tmp_view.buf, sizeof(val_type)*rows*cols);
			}
			return *this;
		} // }}}
		dmat_t transpose() const { // {{{
			dmat_t ret = get_view();
			ret.to_transpose();
			return ret;
		} // }}}

		// In-place functions
		dmat_t& assign(const dmat_t& other) { // {{{
			return assign((val_type)1.0, other);
		} // }}}
		template<typename T>
		dmat_t& assign(T a, const dmat_t& other) { // {{{
			if(a == T(0))
				memset(buf, 0, sizeof(val_type)*rows*cols);
			else if(a == T(1)) {
				if(this == &other)
					return *this;
				if(is_rowmajor()) {
#pragma omp parallel for schedule(static)
					for(size_t r = 0; r < rows; r++)
						for(size_t c = 0; c < cols; c++)
							at(r,c) = other.at(r,c);
				} else {
#pragma omp parallel for schedule (static)
					for(size_t c = 0; c < cols; c++)
						for(size_t r = 0; r < rows; r++)
							at(r,c) = other.at(r,c);
				}
			} else {
				if(is_rowmajor()) {
#pragma omp parallel for schedule(static)
					for(size_t r = 0; r < rows; r++)
						for(size_t c = 0; c < cols; c++)
							at(r,c) = a*other.at(r,c);
				} else {
#pragma omp parallel for schedule(static)
					for(size_t c = 0; c < cols; c++)
						for(size_t r = 0; r < rows; r++)
							at(r,c) = a*other.at(r,c);
				}
			}
			return *this;
		} // }}}
		dmat_t& to_transpose() { // {{{
			std::swap(rows,cols);
			major_type = is_rowmajor()? COLMAJOR: ROWMAJOR;
			init_vec_set();
			return *this;
		} // }}}
		dmat_t& to_rowmajor() {if(is_colmajor()) inv_major(); return *this;}
		dmat_t& to_colmajor() {if(is_rowmajor()) inv_major(); return *this;}
		dmat_t& apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm) { // {{{
			return apply_permutation(row_perm.size()==rows? &row_perm[0]: NULL, col_perm.size()==cols? &col_perm[0] : NULL);
		} // }}}
		dmat_t& apply_permutation(const unsigned *row_perm=NULL, const unsigned *col_perm=NULL) { // {{{
			dmat_t tmp(*this);
			resize(rows,cols);
			for(size_t r = 0; r < rows; r++)
				for(size_t c = 0; c < cols; c++)
					at(r,c) = tmp.at(row_perm? row_perm[r]: r, col_perm? col_perm[c]: c);
			return *this;
		} // }}}

		// IO methods
		void load_from_binary(const char *filename, major_t major_type_=default_major) { // {{{
			FILE *fp = fopen(filename, "rb");
			if(fp == NULL) {
				fprintf(stderr, "Error: can't read the file (%s)!!\n", filename);
				return;
			}
			load_from_binary(fp, major_type_, filename);
			fclose(fp);
		} // }}}
		void load_from_binary(FILE *fp, major_t major_type_=default_major, const char *filename=NULL) { // {{{
			clear_space();
			zero_init();
			size_t rows_, cols_;
			if(fread(&rows_, sizeof(size_t), 1, fp) != 1)
				fprintf(stderr, "Error: wrong input stream in %s.\n", filename);
			if(fread(&cols_, sizeof(size_t), 1, fp) != 1)
				fprintf(stderr, "Error: wrong input stream in %s.\n", filename);

			std::vector<double> tmp(rows_*cols_);
			if(fread(&tmp[0], sizeof(double), rows_*cols_, fp) != rows_*cols_)
				fprintf(stderr, "Error: wrong input stream in %s.\n", filename);
			dmat_t<double> tmp_view(rows_, cols_, &tmp[0], ROWMAJOR);
			major_type = major_type_;
			resize(rows_, cols_);
			for(size_t r = 0; r < rows; r++)
				for(size_t c = 0; c < cols; c++)
					at(r,c) = tmp_view.at(r,c);
			/*
			major_type = major_type_;
			if(major_type_ == ROWMAJOR) {
				resize(rows_, cols_);
				for(size_t idx=0; idx <rows*cols; idx++)
					buf[idx] = (val_type)tmp[idx];
			} else {
				dmat_t tmp_view(rows, cols, &buf[0], ROWMAJOR);
				*this = dmat_t(tmp_view, major_type_);
			}
			*/
		} // }}}
		void save_binary_to_file(const char *filename) { // {{{
			FILE *fp = fopen(filename, "wb");
			if(fp == NULL) {
				fprintf(stderr,"Error: can't open file %s\n", filename);
				exit(1);
			}
			save_binary_to_file(fp);
			fclose(fp);
		} // }}}
		void save_binary_to_file(FILE *fp) { // {{{
			fwrite(&rows, sizeof(size_t), 1, fp);
			fwrite(&cols, sizeof(size_t), 1, fp);
			std::vector<double> tmp(rows*cols);
			size_t idx = 0;
			for(size_t r = 0; r < rows; r++)
				for(size_t c = 0; c < cols; c++)
					tmp[idx++] = (double)at(r,c);
			fwrite(&tmp[0], sizeof(double), tmp.size(), fp);
		} // }}}

		size_t size() const {return rows;}
		void resize(size_t nr_copy, const vec_t &v) { // {{{
			if(is_rowmajor()) {
				size_t rows_ = nr_copy, cols_ = v.size();
				resize(rows_, cols_);
				size_t unit = sizeof(val_type)*v.size();
				for(size_t r = 0; r < rows; r++)
					memcpy(vec_set[r].data(),v.data(),unit);
			} else {
				size_t rows_ = v.size(), cols_ = nr_copy;
				resize(rows_, cols_);
				size_t unit = sizeof(val_type)*v.size();
				for(size_t c = 0; c < cols; c++)
					memcpy(vec_set[c].data(),v.data(),unit);
			}
		} // }}}
		void resize(size_t rows_, size_t cols_) { // {{{
			if(mem_alloc_by_me) {
				if(rows_*cols_ != rows*cols)
					buf = (val_type*) realloc(buf, sizeof(val_type)*rows_*cols_);
			} else {
				buf = (val_type*) malloc(sizeof(val_type)*rows_*cols_);
			}
			mem_alloc_by_me = true;
			rows = rows_; cols = cols_;
			init_vec_set();
		} // }}}
		dmat_t& lazy_resize(size_t rows_, size_t cols_, major_t major_type_=0) { // {{{
			if(is_view() && rows_*cols_==rows*cols &&
					(major_type_ == 0 || major_type==major_type_))
				reshape(rows_,cols_);
			else {
				if(major_type_!=0) major_type = major_type_;
				resize(rows_, cols_);
			}
			return *this;
		} // }}}
		dmat_t& reshape(size_t rows_, size_t cols_) { // {{{
			assert(rows_*cols_ == rows*cols);
			if(rows_ != rows || cols != cols) {
				rows = rows_; cols = cols_;
				init_vec_set();
			}
			return *this;
		} // }}}
		inline val_type& at(size_t r, size_t c) {return is_rowmajor()? buf[r*cols+c] : buf[c*rows+r];}
		inline const val_type& at(size_t r, size_t c) const {return is_rowmajor()? buf[r*cols+c] : buf[c*rows+r];}
		vec_t& operator[](size_t idx) {return vec_set[idx];}
		const vec_t& operator[](size_t idx) const {return vec_set[idx];}
		val_type* data() {return buf;}
		const val_type* data() const {return buf;}

		void print_mat(const char *str="", FILE *fp=stdout) const { // {{{
			fprintf(fp, "===>%s<===\n", str);
			fprintf(fp, "rows %ld cols %ld mem_alloc_by_me %d row_major %d buf %p\n",
					rows, cols, mem_alloc_by_me, is_rowmajor(), buf);
			for(size_t r = 0; r < rows; r++) {
				for(size_t c = 0; c < cols; c++)
					fprintf(fp, "%g ", at(r,c));
				fprintf(fp, "\n");
			}
		} // }}}
}; // }}}

// Identity Matrix
template<typename val_type>
class eye_t : public gmat_t<val_type> { // {{{
	public:
		// size_t rows, cols; inherited from gmat_t
		using gmat_t<val_type>::rows;
		using gmat_t<val_type>::cols;
		eye_t (size_t rows_ = 0): gmat_t<val_type>(rows_,rows_){}
		bool is_identity() const {return true;}
}; // }}}

// Sparse matrix format CSC & CSR
template<typename val_type>
class smat_t : public gmat_t<val_type> { // {{{
	private:
		bool mem_alloc_by_me;
		bool read_from_binary;
		unsigned char* binary_buf;
		size_t binary_buf_len;
		const static int HeaderSize =
			sizeof(size_t)+sizeof(size_t)+sizeof(size_t)+sizeof(size_t);
		void zero_init();
		void allocate_space(size_t rows_, size_t cols_, size_t nnz_);
		void csr_to_csc();
		void csc_to_csr();
		void update_max_nnz();

	public: // static methods
		static smat_t rand(rng_t &rng, size_t m, size_t n, double sparsity=0.01, double lower=0.0, double upper=1.0) { // {{{
			if(lower > upper) lower = upper;
			smat_t ret;
			size_t nnz_ = (size_t)(m*n*sparsity);
			ret.allocate_space(m, n, nnz_);
			for(size_t idx = 0; idx < nnz_; idx++) {
				ret.val_t[idx] = rng.uniform(lower, upper);
				ret.col_idx[idx] = rng.randint(0, n-1);
				ret.row_ptr[rng.randint(1, m)] += 1;
			}
			for(size_t i = 1; i <= m; i++)
				ret.row_ptr[i] += ret.row_ptr[i-1];
			ret.csr_to_csc();
			ret.update_max_nnz();
			return ret;
		} // }}}
		static smat_t randn(rng_t &rng, size_t m, size_t n, double sparsity=0.01, double mean=0.0, double std=1.0) { // {{{
			smat_t ret;
			size_t nnz_ = (size_t)(m*n*sparsity);
			ret.allocate_space(m, n, nnz_);
			for(size_t idx = 0; idx < nnz_; idx++) {
				ret.val_t[idx] = (val_type)rng.normal(mean, std);
				ret.col_idx[idx] = rng.randint(0, n-1);
				ret.row_ptr[rng.randint(1,m)] += 1;
			}
			for(size_t i = 1; i <= m; i++)
				ret.row_ptr[i] += ret.row_ptr[i-1];
			ret.csr_to_csc();
			ret.update_max_nnz();
			return ret;
		} // }}}

	public:
		//size_t rows, cols; // inherited from gmat_t
		using gmat_t<val_type>::rows;
		using gmat_t<val_type>::cols;
		size_t nnz, max_row_nnz, max_col_nnz;
		val_type *val, *val_t;
		size_t *col_ptr, *row_ptr;
		unsigned *row_idx, *col_idx;

		// filetypes for loading smat_t
		enum format_t {TXT=0, PETSc=1, BINARY=2, COMPRESSION=3};

		// Default Constructor
		smat_t() {zero_init();}
		// Copy Constructor
		smat_t(const smat_t& m) {zero_init(); *this = m;}
		smat_t(const dmat_t<val_type>& m) { // {{{
			zero_init();
			dmat_iterator_t<val_type> entry_it(m);
			load_from_iterator(m.rows, m.cols, entry_it.get_nnz(), &entry_it);
		} //}}}
		smat_t(const eye_t<val_type>& eye) { // {{{
			zero_init();
			allocate_space(eye.rows, eye.rows, 0);
			for(size_t i = 0; i < eye.rows; i++) {
				row_ptr[i+1] = i+1;
				col_idx[i] = i;
				val_t[i] = (val_type)1;
			}
			for(size_t j = 0; j < eye.cols; j++) {
				col_ptr[j+1] = j+1;
				row_idx[j] = j;
				val[j] = (val_type)1;
			}
		} // }}}
		smat_t(size_t rows_, size_t cols_, size_t nnz_=0){ // {{{
			zero_init();
			allocate_space(rows_, cols_, nnz_);
		} // }}}
		// Copy Assignment
		smat_t& operator=(const smat_t& other) { // {{{
			if(this == &other) return *this;
			if(mem_alloc_by_me) clear_space();
			if(other.is_view()) // for view
				memcpy(this, &other, sizeof(smat_t));
			else { // deep copy
				*this = other.get_view();
				grow_body();
			}
			return *this;
		} // }}}
#if defined(CPP11)
		// Move Constructor
		smat_t(smat_t&& m){zero_init(); *this = std::move(m);}
		// Move Assignment
		smat_t& operator=(smat_t&& other) { // {{{
			if(this == &other) return *this;
			clear_space();
			memcpy(this, &other, sizeof(smat_t));
			other.zero_init();
			return *this;
		} // }}}
#endif
		// Destructor
		~smat_t(){ clear_space();}

		bool is_view() const {return mem_alloc_by_me==false;}
		bool is_sparse() const {return true;}
		void clear_space();
		smat_t get_view() const;
		smat_t& grow_body();

		smat_t transpose() const; // return a transpose view
		//const smat_t transpose() const; // return a transpose view

		// In-place functions
		smat_t& to_transpose(); // return a transpose view
		smat_t& apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm);
		smat_t& apply_permutation(const unsigned *row_perm=NULL, const unsigned *col_perm=NULL);

		smat_subset_iterator_t<val_type> row_subset_it(const std::vector<unsigned> &subset) const;
		smat_subset_iterator_t<val_type> row_subset_it(const unsigned *subset, int subset_size) const;
		smat_subset_iterator_t<val_type> col_subset_it(const std::vector<unsigned> &subset) const;
		smat_subset_iterator_t<val_type> col_subset_it(const unsigned *subset, int subset_size) const;
		smat_t row_subset(const std::vector<unsigned> &subset) const;
		smat_t row_subset(const unsigned *subset, int subset_size) const;

		size_t nnz_of_row(unsigned i) const {return (row_ptr[i+1]-row_ptr[i]);}
		size_t nnz_of_col(unsigned i) const {return (col_ptr[i+1]-col_ptr[i]);}

		// smat-vector multiplication
		val_type* Xv(const val_type* v, val_type* Xv) const;
		dvec_t<val_type>& Xv(const dvec_t<val_type>& v, dvec_t<val_type>& Xv) const;
		val_type* XTu(const val_type* u, val_type* XTu) const;
		dvec_t<val_type>& XTu(const dvec_t<val_type>& u, dvec_t<val_type>& XTu) const;

		// IO methods
		void load_from_iterator(size_t _rows, size_t _cols, size_t _nnz, entry_iterator_t<val_type>* entry_it);
		void load(size_t _rows, size_t _cols, size_t _nnz, const char *filename, format_t fmt);
		void load_from_PETSc(const char *filename);
		void load_from_PETSc(FILE *fp, const char *filename=NULL);
		void save_PETSc_to_file(const char *filename) const;
		void save_PETSc_to_file(FILE *fp) const;
		void load_from_binary(const char *filename);
		void save_binary_to_file(const char *filename) const ;

		// used for MPI verions
		void from_mpi(){ // {{{
			mem_alloc_by_me = true;
			max_col_nnz = 0;
			for(size_t c = 0; c < cols; c++)
				max_col_nnz = std::max(max_col_nnz, nnz_of_col(c));
		} // }}}
		val_type get_global_mean() const;
		void remove_bias(val_type bias=0);

		void print_mat(const char *str="", FILE *fp=stdout) const { // {{{
			fprintf(fp, "===>%s<===\n", str);
			fprintf(fp, "rows,cols,nnz = %lu, %lu, %lu\n", rows, cols, nnz);
			fprintf(fp, "col_ptr, row_idx, val = %p, %p, %p\n", col_ptr, row_idx, val);
			fprintf(fp, "row_ptr, col_idx, val_t = %p, %p, %p\n", row_ptr, col_idx, val_t);
			fprintf(fp, "mem_alloc_by_me = %d\n", mem_alloc_by_me);
			fprintf(fp, "read_from_binary = %d\n", read_from_binary);
		} // }}}
}; // }}}

// Lapack and Blas support {{{
#ifdef _WIN32
#define ddot_ ddot
#define sdot_ sdot
#define daxpy_ daxpy
#define saxpy_ saxpy
#define dcopy_ dcopy
#define scopy_ scopy
#define dgemm_ dgemm
#define sgemm_ sgemm
#define dposv_ dposv
#define sposv_ sposv
#define dgesdd_ dgesdd
#define sgesdd_ sgesdd
#endif

extern "C" {

	double ddot_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
	float sdot_(ptrdiff_t *, float *, ptrdiff_t *, float *, ptrdiff_t *);

	ptrdiff_t dscal_(ptrdiff_t *, double *, double *, ptrdiff_t *);
	ptrdiff_t sscal_(ptrdiff_t *, float *, float *, ptrdiff_t *);

	ptrdiff_t daxpy_(ptrdiff_t *, double *, double *, ptrdiff_t *, double *, ptrdiff_t *);
	ptrdiff_t saxpy_(ptrdiff_t *, float *, float *, ptrdiff_t *, float *, ptrdiff_t *);

	double dcopy_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
	float scopy_(ptrdiff_t *, float *, ptrdiff_t *, float *, ptrdiff_t *);

	void dgemm_(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, double *alpha, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, double *beta, double *c, ptrdiff_t *ldc);
	void sgemm_(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, float *alpha, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, float *beta, float *c, ptrdiff_t *ldc);

	int dposv_(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, ptrdiff_t *info);
	int sposv_(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, ptrdiff_t *info);

	void dgesdd_(char* jobz, ptrdiff_t* m, ptrdiff_t* n, double* a, ptrdiff_t* lda, double* s, double* u, ptrdiff_t* ldu, double* vt, ptrdiff_t* ldvt, double* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info);
	void sgesdd_(char* jobz, ptrdiff_t* m, ptrdiff_t* n, float* a, ptrdiff_t* lda, float* s, float* u, ptrdiff_t* ldu, float* vt, ptrdiff_t* ldvt, float* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info);

}

template<typename val_type> val_type dot(ptrdiff_t *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
template<> inline double dot(ptrdiff_t *len, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return ddot_(len,x,xinc,y,yinc);}
template<> inline float dot(ptrdiff_t *len, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return sdot_(len,x,xinc,y,yinc);}

template<typename val_type> val_type scal(ptrdiff_t *, val_type *, val_type *, ptrdiff_t *);
template<> inline double scal(ptrdiff_t *len, double *a, double *x, ptrdiff_t *xinc) { return dscal_(len,a,x,xinc);}
template<> inline float scal(ptrdiff_t *len, float *a,  float *x, ptrdiff_t *xinc) { return sscal_(len,a,x,xinc);}

template<typename val_type> ptrdiff_t axpy(ptrdiff_t *, val_type *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
template<> inline ptrdiff_t axpy(ptrdiff_t *len, double *alpha, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return daxpy_(len,alpha,x,xinc,y,yinc);};
template<> inline ptrdiff_t axpy(ptrdiff_t *len, float *alpha, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return saxpy_(len,alpha,x,xinc,y,yinc);};

template<typename val_type> val_type copy(ptrdiff_t *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
template<> inline double copy(ptrdiff_t *len, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return dcopy_(len,x,xinc,y,yinc);}
template<> inline float copy(ptrdiff_t *len, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return scopy_(len,x,xinc,y,yinc);}

template<typename val_type> void gemm(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, val_type *alpha, val_type *a, ptrdiff_t *lda, val_type *b, ptrdiff_t *ldb, val_type *beta, val_type *c, ptrdiff_t *ldc);
template<> inline void gemm(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, double *alpha, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, double *beta, double *c, ptrdiff_t *ldc) { dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
template<> inline void gemm<float>(char *transa, char *transb, ptrdiff_t *m, ptrdiff_t *n, ptrdiff_t *k, float *alpha, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, float *beta, float *c, ptrdiff_t *ldc) { sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

template<typename val_type> int posv(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, val_type *a, ptrdiff_t *lda, val_type *b, ptrdiff_t *ldb, ptrdiff_t *info);
template<> inline int posv(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, double *a, ptrdiff_t *lda, double *b, ptrdiff_t *ldb, ptrdiff_t *info) { return dposv_(uplo, n, nrhs, a, lda, b, ldb, info); }
template<> inline int posv(char *uplo, ptrdiff_t *n, ptrdiff_t *nrhs, float *a, ptrdiff_t *lda, float *b, ptrdiff_t *ldb, ptrdiff_t *info) { return sposv_(uplo, n, nrhs, a, lda, b, ldb, info); }

template<typename val_type> void gesdd(char* jobz, ptrdiff_t* m, ptrdiff_t* n, val_type* a, ptrdiff_t* lda, val_type* s, val_type* u, ptrdiff_t* ldu, val_type* vt, ptrdiff_t* ldvt, val_type* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info);
template<> inline void gesdd(char* jobz, ptrdiff_t* m, ptrdiff_t* n, double* a, ptrdiff_t* lda, double* s, double* u, ptrdiff_t* ldu, double* vt, ptrdiff_t* ldvt, double* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info) { return dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info); }
template<> inline void gesdd(char* jobz, ptrdiff_t* m, ptrdiff_t* n, float* a, ptrdiff_t* lda, float* s, float* u, ptrdiff_t* ldu, float* vt, ptrdiff_t* ldvt, float* work, ptrdiff_t* lwork, ptrdiff_t* iwork, ptrdiff_t* info) { return sgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info); }
// }}}

// <x,y>
template<typename val_type>
val_type do_dot_product(const val_type *x, const val_type *y, size_t size) { // {{{
	val_type *xx = const_cast<val_type*>(x);
	val_type *yy = const_cast<val_type*>(y);
	ptrdiff_t inc = 1;
	ptrdiff_t len = (ptrdiff_t) size;
	return dot(&len, xx, &inc, yy, &inc);
} // }}}
template<typename val_type>
val_type do_dot_product(const dvec_t<val_type> &x, const dvec_t<val_type> &y) { // {{{
	assert(x.size() == y.size());
	return do_dot_product(x.data(), y.data(), x.size());
} // }}}
template<typename val_type>
val_type do_dot_product(const dmat_t<val_type> &x, const dmat_t<val_type> &y) { // {{{
	assert(x.rows == y.rows && x.cols == y.cols);
	if((x.is_rowmajor() && y.is_rowmajor()) || (x.is_colmajor() && y.is_colmajor()))
		return do_dot_product(x.data(), y.data(), x.rows*x.cols);
	else {
		val_type ret = 0.0;
		const dmat_t<val_type> &xx = (x.rows > x.cols) ? x : x.transpose();
		const dmat_t<val_type> &yy = (y.rows > y.cols) ? y : y.transpose();
#pragma omp parallel for schedule(static) reduction(+:ret)
		for(size_t i = 0; i < xx.rows; i++) {
			double ret_local = 0.0;
			for(size_t j = 0; j < xx.cols; j++)
				ret_local += xx.at(i,j)*yy.at(i,j);
			ret += ret_local;
		}
		return (val_type)ret;
	}
} // }}}

// y = alpha*x + y
template<typename val_type, typename T>
void do_axpy(T alpha, const val_type *x, val_type *y, size_t size) { // {{{
	if(alpha == 0) return;
	val_type alpha_ = (val_type)alpha;
	ptrdiff_t inc = 1;
	ptrdiff_t len = (ptrdiff_t) size;
	val_type *xx = const_cast<val_type*>(x);
	axpy(&len, &alpha_, xx, &inc, y, &inc);
} // }}}
template<typename val_type, typename T>
void do_axpy(T alpha, const dvec_t<val_type> &x, dvec_t<val_type> &y) { // {{{
	do_axpy(alpha, x.data(), y.data(), x.size());
} // }}}
template<typename val_type, typename T>
void do_axpy(T alpha, const dmat_t<val_type> &x, dmat_t<val_type> &y) { // {{{
	assert(x.rows == y.rows && x.cols == y.cols);
	if((x.is_rowmajor() && y.is_rowmajor()) || (x.is_colmajor() && y.is_colmajor()))
		do_axpy(alpha, x.data(), y.data(), x.rows*x.cols);
	else {
		if(x.rows > x.cols) {
#pragma omp parallel for schedule(static)
			for(size_t i = 0; i < x.rows; i++)
				for(size_t j = 0; j < x.cols; j++)
					y.at(i,j) += alpha*x.at(i,j);
		} else {
#pragma omp parallel for schedule(static)
			for(size_t j = 0; j < x.cols; j++)
				for(size_t i = 0; i < x.rows; i++)
					y.at(i,j) += alpha*x.at(i,j);
		}
	}
} // }}}

// x *= alpha
template<typename val_type, typename T>
void do_scale(T alpha, val_type *x, size_t size) { // {{{
	if(alpha == 0.0) {
		memset(x, 0, sizeof(val_type)*size);
	} else if (alpha == 1.0) {
			return;
	} else {
		val_type alpha_minus_one = (val_type)(alpha-1);
		do_axpy(alpha_minus_one, x, x, size);
	}
} // }}}
template<typename val_type, typename T>
val_type do_scale(T alpha, dvec_t<val_type> &x) { // {{{
	do_scale(alpha, x.data(), x.size());
} // }}}
template<typename val_type, typename T>
val_type do_scale(T alpha, dmat_t<val_type> &x) { // {{{
	do_scale(alpha, x.data(), x.rows*x.cols);
} // }}}

// y = x
template<typename val_type>
void do_copy(const val_type *x, val_type *y, size_t size) { // {{{
	if(x == y) return;
	ptrdiff_t inc = 1;
	ptrdiff_t len = (ptrdiff_t) size;
	val_type *xx = const_cast<val_type*>(x);
	copy(&len, xx, &inc, y, &inc);
} // }}}

// A, B, C are stored in column major!
template<typename val_type, typename T1, typename T2>
void dmat_x_dmat_colmajor(T1 alpha, const val_type *A, bool trans_A, const val_type *B, bool trans_B, T2 beta, val_type *C, size_t m, size_t n, size_t k) { // {{{
	ptrdiff_t mm = (ptrdiff_t)m, nn = (ptrdiff_t)n, kk = (ptrdiff_t)k;
	ptrdiff_t lda = trans_A? kk:mm, ldb = trans_B? nn:kk, ldc = mm;
	char transpose = 'T', notranspose = 'N';
	char *transa = trans_A? &transpose: &notranspose;
	char *transb = trans_B? &transpose: &notranspose;
	val_type alpha_ = (val_type) alpha;
	val_type beta_ = (val_type) beta;
	val_type *AA = const_cast<val_type*>(A);
	val_type *BB = const_cast<val_type*>(B);
	gemm(transa, transb, &mm, &nn, &kk, &alpha_, AA, &lda, BB, &ldb, &beta_, C, &ldc);
} // }}}

// C = alpha*A*B + beta*C
// C : m * n, k is the dimension of the middle
// A, B, C are stored in row major!
template<typename val_type, typename T1, typename T2>
void dmat_x_dmat(T1 alpha, const val_type *A, bool trans_A, const val_type *B, bool trans_B, T2 beta, val_type *C, size_t m, size_t n, size_t k) { // {{{
	dmat_x_dmat_colmajor(alpha, B, trans_B, A, trans_A, beta, C, n, m, k);
} //}}}

// C = A'*B
// C : m*n, k is the dimension of the middle
// A, B, C are stored in row major!
template<typename val_type>
void dmat_trans_x_dmat(const val_type *A, const val_type *B, val_type *C, size_t m, size_t n, size_t k) { // {{{
	bool trans = true; dmat_x_dmat(val_type(1.0), A, trans, B, !trans, val_type(0.0), C, m, n, k);
} // }}}

// C=A*B
// A, B, C are stored in row major!
template<typename val_type>
void dmat_x_dmat(const val_type *A, const val_type *B, val_type *C, size_t m, size_t n, size_t k) { // {{{
	bool trans = true; dmat_x_dmat(val_type(1.0), A, !trans, B, !trans, val_type(0.0), C, m, n, k);
} // }}}

// Input: an n*k row-major matrix H
// Output: an k*k matrix H^TH
template<typename val_type>
void doHTH(const val_type *H, val_type *HTH, size_t n, size_t k) { // {{{
	bool transpose = true;
	dmat_x_dmat_colmajor(val_type(1.0), H, !transpose, H, transpose, val_type(0.0), HTH, k, k, n);
} // }}}

// Solve Ax = b, A is symmetric positive definite, b is overwritten with the result x
// A will be modifed by internal Lapack. Make copy when necessary
template<typename val_type>
bool ls_solve_chol(val_type *A, val_type *b, size_t n) { // {{{
  ptrdiff_t nn=n, lda=n, ldb=n, nrhs=1, info=0;
  char uplo = 'U';
  posv(&uplo, &nn, &nrhs, A, &lda, b, &ldb, &info);
  return (info == 0);
} // }}}

// Solve AX = B, A is symmetric positive definite, B is overwritten with the result X
// A is a m-by-m matrix, while B is a m-by-n matrix stored in col_major
// A will be modifed by internal Lapack. Make copy when necessary
template<typename val_type>
bool ls_solve_chol_matrix_colmajor(val_type *A, val_type *B, size_t m, size_t n = size_t(0)) { // {{{
  ptrdiff_t mm=m, lda=m, ldb=m, nrhs=n, info=0;
  char uplo = 'U';
  posv(&uplo, &mm, &nrhs, A, &lda, B, &ldb, &info);
  return (info == 0);
} // }}}

// Functions for dmat_t type
// C = alpha*A*B + beta*C
// C : m * n, k is the dimension of the middle
template<typename val_type, typename T1, typename T2>
dmat_t<val_type>& dmat_x_dmat(T1 alpha, const dmat_t<val_type>& A, const dmat_t<val_type>& B, T2 beta, dmat_t<val_type>& C) { // {{{
	assert(A.cols == B.rows);
	dmat_t<val_type> AA = A.get_view(), BB = B.get_view();
	C.lazy_resize(AA.rows, BB.cols);
	if (C.is_rowmajor()) {
		bool trans_A = A.is_rowmajor()? false : true;
		bool trans_B = B.is_rowmajor()? false : true;
		dmat_x_dmat(alpha, AA.data(), trans_A, BB.data(), trans_B, beta, C.data(), C.rows, C.cols, A.cols);
	} else {
		bool trans_A = A.is_colmajor()? false : true;
		bool trans_B = B.is_colmajor()? false : true;
		dmat_x_dmat_colmajor(alpha, AA.data(), trans_A, BB.data(), trans_B, beta, C.data(), C.rows, C.cols, A.cols);
	}
	return C;
} // }}}
// C=A*B
template<typename val_type>
dmat_t<val_type>& dmat_x_dmat(const dmat_t<val_type>& A, const dmat_t<val_type>& B, dmat_t<val_type>& C) { // {{{
	return dmat_x_dmat(val_type(1.0), A, B, val_type(0.0), C);
} // }}}
template<typename val_type>
dmat_t<val_type> operator*(const dmat_t<val_type>& A, const dmat_t<val_type>& B) { // {{{
	dmat_t<val_type> C(A.rows,B.cols);
	dmat_x_dmat(A,B,C);
	return C;
} // }}}
// Solve AX = B, A is symmetric positive definite, return X
template<typename val_type>
dmat_t<val_type> ls_solve_chol(const dmat_t<val_type>& A, const dmat_t<val_type>& B) { // {{{
	dmat_t<val_type> X(B, COLMAJOR); X.grow_body();
	dmat_t<val_type> AA(A); AA.grow_body();
	if(ls_solve_chol_matrix_colmajor(AA.data(), X.data(), AA.rows, X.cols) == false)
		fprintf(stderr, "error to apply ls_solve_cho_matrix_colmajor");
	return X;
} // }}}
// SVD [U S V] = SVD(A),
template<typename val_type>
class svd_solver_t { // {{{
	private:
		char jobz;
		ptrdiff_t mm, nn, min_mn, max_mn, lda, ldu, ldvt, lwork1, lwork2, lwork, info;
		std::vector<val_type> u_buf, v_buf, s_buf, work;
		std::vector<ptrdiff_t> iwork;
		size_t k;

		void prepare_parameter(const dmat_t<val_type>& A, dmat_t<val_type>& U, dvec_t<val_type>& S, dmat_t<val_type>& V, bool reduced) { // {{{
			k = std::min(A.rows, A.cols);
			mm = (ptrdiff_t)A.rows;
			nn = (ptrdiff_t)A.cols;
			min_mn = std::min(mm,nn);
			max_mn = std::max(mm,nn);
			lda = mm;
			ldu = mm;
			ldvt = reduced? min_mn : nn;
			lwork1 = 3*min_mn*min_mn + std::max(max_mn, 4*min_mn*min_mn + 4*min_mn);
			lwork2 = 3*min_mn + std::max(max_mn, 4*min_mn*min_mn + 3*min_mn + max_mn);
			lwork = 2 * std::max(lwork1, lwork2);  // due to differences between lapack 3.1 and 3.4
			info = 0;
			work.resize(lwork);
			iwork.resize((size_t)(8*min_mn));
			if(!S.is_view() || S.size() != k)
				S.resize(k);
			if(reduced) {
				jobz = 'S';
				U.lazy_resize(A.rows, k, COLMAJOR);
				V.lazy_resize(A.cols, k, ROWMAJOR);
			} else {
				jobz = 'A';
				U.lazy_resize(A.rows, A.rows, COLMAJOR);
				V.lazy_resize(A.cols, A.cols, ROWMAJOR);
			}
		} // }}}
	public:
		svd_solver_t() {}
		bool solve(const dmat_t<val_type>& A, dmat_t<val_type>& U, dvec_t<val_type>& S, dmat_t<val_type>& V, bool reduced=true) { // {{{
			if(A.is_rowmajor())
				return solve(A.transpose(), V, S, U, reduced);
			 else {
				 dmat_t<val_type> AA(A.get_view());
				 prepare_parameter(AA, U, S, V, reduced);
#if defined(CPP11)
				gesdd(&jobz, &mm, &nn, AA.data(), &lda,
						S.data(), U.data(), &ldu, V.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
#else
				gesdd(&jobz, &mm, &nn, AA.data(), &lda,
						S.data(), U.data(), &ldu, V.data(), &ldvt, &work[0], &lwork, &iwork[0], &info);
#endif
				return (info == 0);
			}
		} // }}}
}; // }}}
template<typename val_type>
void svd(const dmat_t<val_type>& A, dmat_t<val_type>& U, dvec_t<val_type>& S, dmat_t<val_type>& V, bool reduced=true) { // {{{
	svd_solver_t<val_type> solver;
	solver.solve(A, U, S, V, reduced);
} // }}}

template<typename val_type>
smat_t<val_type> sprand(size_t m, size_t n, double sparsity) { // {{{
	static rng_t rng;
	return smat_t<val_type>::rand(rng, m, n, sparsity);
} // }}}
template<typename val_type>
smat_t<val_type> sprandn(size_t m, size_t n, double sparsity) { // {{{{
	static rng_t rng;
	return smat_t<val_type>::randn(rng, m, n, sparsity);
} // }}}

template<typename val_type>
dmat_t<val_type> drand(size_t m, size_t n, major_t major_type_=default_major) { // {{{
	static rng_t rng;
	return dmat_t<val_type>::rand(rng, m, n, 0.0, 1.0, major_type_ );
} // }}}
template<typename val_type>
dmat_t<val_type> drandn(size_t m, size_t n, major_t major_type_=default_major) { // {{{{
	static rng_t rng;
	return dmat_t<val_type>::randn(rng, m, n, 0.0, 1.0, major_type_);
} // }}}


/*-------------- Iterators -------------------*/

template<typename val_type>
class entry_t{ // {{{
	public:
		unsigned i, j; val_type v, weight;
		entry_t(int ii=0, int jj=0, val_type vv=0, val_type ww=1.0): i(ii), j(jj), v(vv), weight(ww){}
}; // }}}

template<typename val_type>
class entry_iterator_t { // {{{
	public:
		size_t nnz;
		virtual entry_t<val_type> next() = 0;
}; // }}}

#define MAXLINE 10240
// Iterator for files with (i,j,v) tuples
template<typename val_type>
class file_iterator_t: public entry_iterator_t<val_type> { // {{{
	public:
		file_iterator_t(size_t nnz_, const char* filename, size_t start_pos=0);
		~file_iterator_t(){ if (fp) fclose(fp); }
		entry_t<val_type> next();
	private:
		size_t nnz;
		FILE *fp;
		char line[MAXLINE];
}; // }}}

// smat_t iterator
template<typename val_type>
class smat_iterator_t: public entry_iterator_t<val_type> { // {{{
	public:
		//enum {ROWMAJOR, COLMAJOR};
		// major: smat_iterator_t<val_type>::ROWMAJOR or smat_iterator_t<val_type>::COLMAJOR
		smat_iterator_t(const smat_t<val_type>& M, major_t major = ROWMAJOR);
		~smat_iterator_t() {}
		entry_t<val_type> next();
	private:
		size_t nnz;
		unsigned *col_idx;
		size_t *row_ptr;
		val_type *val_t;
		size_t rows, cols, cur_idx;
		size_t cur_row;
}; // }}}

// smat_t subset iterator
template<typename val_type>
class smat_subset_iterator_t: public entry_iterator_t<val_type> { // {{{
	public:
		//enum {ROWMAJOR, COLMAJOR};
		// major: smat_iterator_t<val_type>::ROWMAJOR or smat_iterator_t<val_type>::COLMAJOR
		smat_subset_iterator_t(const smat_t<val_type>& M, const unsigned *subset, size_t size, bool remapping=false, major_t major = ROWMAJOR);
		~smat_subset_iterator_t() {}
		size_t get_nnz() {return nnz;}
		size_t get_rows() {return major==ROWMAJOR? remapping? subset.size(): rows: rows;}
		size_t get_cols() {return major==ROWMAJOR? cols: remapping? subset.size():cols;}
		entry_t<val_type> next();
	private:
		size_t nnz;
		unsigned *col_idx;
		size_t *row_ptr;
		val_type *val_t;
		size_t rows, cols, cur_idx;
		size_t cur_row;
		std::vector<unsigned>subset;
		major_t major;
		bool remapping;
}; // }}}

// dmat_t iterator
template<typename val_type>
class dmat_iterator_t: public entry_iterator_t<val_type> { // {{{
	public:
		dmat_iterator_t(const dmat_t<val_type>& M, double threshold=1e-12) : M(M), nnz(M.rows*M.cols), rows(M.rows), cols(M.cols), threshold(fabs(threshold)) { // {{{
			cur_row = 0;
			cur_col = 0;
			nnz = 0;
			bool find_firstnz = true;
			for(size_t i = 0; i < rows; i++)
				for(size_t j = 0; j < cols; j++)
					if(fabs((double)M.at(i,j)) >= threshold) {
						if(find_firstnz) {
							cur_row = i;
							cur_col = j;
							find_firstnz = false;
						}
						nnz++ ;
					}
		//	printf("cur_row %ld cur_col %ld nnz %ld\n", cur_row, cur_col, nnz);
		} // }}}
		~dmat_iterator_t() {}
		entry_t<val_type> next() { // {{{
			entry_t<val_type> entry(cur_row, cur_col, M.at(cur_row, cur_col));
			do {
				cur_col += 1;
				if(cur_col == cols) {
					cur_row += 1;
					cur_col = 0;
				}
			} while(fabs((double)M.at(cur_row, cur_col)) <= threshold );
			return entry;
		} // }}}
		size_t get_nnz() const {return nnz;}
	private:
		size_t nnz;
		const dmat_t<val_type>& M;
		size_t rows, cols, cur_row, cur_col;
		double threshold;
}; // }}}

// -------------- Implementation --------------
template<typename val_type>
inline void smat_t<val_type>::zero_init() { // {{{
	mem_alloc_by_me = false;
	read_from_binary = false;
	val=val_t=NULL;
	col_ptr=row_ptr=NULL;
	row_idx=col_idx=NULL;
	rows=cols=nnz=max_col_nnz=max_row_nnz=0;
} // }}}

template<typename val_type>
void smat_t<val_type>::allocate_space(size_t rows_, size_t cols_, size_t nnz_) { //  {{{
	if(mem_alloc_by_me)
		clear_space();
	rows = rows_; cols = cols_; nnz = nnz_;
	val = MALLOC(val_type, nnz); val_t = MALLOC(val_type, nnz);
	row_idx = MALLOC(unsigned, nnz); col_idx = MALLOC(unsigned, nnz);
	row_ptr = MALLOC(size_t, rows+1); col_ptr = MALLOC(size_t, cols+1);
	memset(row_ptr,0,sizeof(size_t)*(rows+1));
	memset(col_ptr,0,sizeof(size_t)*(cols+1));
	mem_alloc_by_me = true;
} // }}}

template<typename val_type>
void smat_t<val_type>::clear_space() { // {{{
	if(mem_alloc_by_me) {
		if(read_from_binary)
			free(binary_buf);
		else {
			if(val)free(val); if(val_t)free(val_t);
			if(row_ptr)free(row_ptr);if(row_idx)free(row_idx);
			if(col_ptr)free(col_ptr);if(col_idx)free(col_idx);
		}
	}
	zero_init();
} // }}}

template<typename val_type>
smat_t<val_type> smat_t<val_type>::get_view() const { // {{{
	if(is_view())
		return *this;
	else {
		smat_t tmp;
		memcpy(&tmp, this, sizeof(smat_t));
		tmp.mem_alloc_by_me = false;
		return tmp;
	}
} // }}}

template<typename val_type>
smat_t<val_type>& smat_t<val_type>::grow_body() { // {{{
	if(is_view()) {
		smat_t tmp = *this; // a copy of the view
		col_ptr = MALLOC(size_t, cols+1); memcpy(col_ptr, tmp.col_ptr, sizeof(size_t)*cols+1);
		row_idx = MALLOC(unsigned, nnz); memcpy(row_idx, tmp.row_idx, sizeof(unsigned)*nnz);
		val = MALLOC(val_type, nnz); memcpy(val, tmp.val, sizeof(val_type)*nnz);
		row_ptr = MALLOC(size_t, rows+1); memcpy(row_ptr, tmp.row_ptr, sizeof(size_t)*rows+1);
		col_idx = MALLOC(unsigned, nnz); memcpy(col_idx, tmp.col_idx, sizeof(unsigned)*nnz);
		val_t = MALLOC(val_type, nnz); memcpy(val_t, tmp.val_t, sizeof(val_type)*nnz);
		mem_alloc_by_me = true;
	}
	return *this;
} // }}}

template<typename val_type>
smat_t<val_type> smat_t<val_type>::transpose() const{ // {{{
	smat_t<val_type> mt = get_view().to_transpose();
	/*
	mt.cols = rows; mt.rows = cols; mt.nnz = nnz;
	mt.val = val_t; mt.val_t = val;
	mt.col_ptr = row_ptr; mt.row_ptr = col_ptr;
	mt.col_idx = row_idx; mt.row_idx = col_idx;
	mt.max_col_nnz=max_row_nnz; mt.max_row_nnz=max_col_nnz;
	*/
	return mt;
} // }}}

template<typename val_type>
smat_t<val_type>& smat_t<val_type>::to_transpose() { // {{{
	std::swap(rows,cols);
	std::swap(val,val_t);
	std::swap(row_ptr,col_ptr);
	std::swap(row_idx,col_idx);
	std::swap(max_col_nnz, max_row_nnz);
	return *this;
} // }}}

template<typename val_type>
smat_t<val_type>& smat_t<val_type>::apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm) { // {{{
	apply_permutation(row_perm.size()==rows? &row_perm[0]: NULL, col_perm.size()==cols? &col_perm[0]: NULL);
} // }}}

template<typename val_type>
smat_t<val_type>& smat_t<val_type>::apply_permutation(const unsigned *row_perm, const unsigned *col_perm) { // {{{
	if(row_perm!=NULL) {
		for(size_t idx = 0; idx < nnz; idx++) row_idx[idx] = row_perm[row_idx[idx]];
		csc_to_csr();
		csr_to_csc();
	}
	if(col_perm!=NULL) {
		for(size_t idx = 0; idx < nnz; idx++) col_idx[idx] = col_perm[col_idx[idx]];
		csr_to_csc();
		csc_to_csr();
	}
} // }}}

template<typename val_type>
smat_subset_iterator_t<val_type> smat_t<val_type>::row_subset_it(const std::vector<unsigned> &subset) const { // {{{
	return row_subset_it(&subset[0], (int)subset.size());
} // }}}

template<typename val_type>
smat_subset_iterator_t<val_type> smat_t<val_type>::row_subset_it(const unsigned *subset, int subset_size) const { // {{{
	return smat_subset_iterator_t<val_type> (*this, subset, subset_size);
} // }}}

template<typename val_type>
smat_subset_iterator_t<val_type> smat_t<val_type>::col_subset_it(const std::vector<unsigned> &subset) const { // {{{
	return col_subset_it(&subset[0], (int)subset.size());
} // }}}

template<typename val_type>
smat_subset_iterator_t<val_type> smat_t<val_type>::col_subset_it(const unsigned *subset, int subset_size) const { // {{{
	bool remmapping = false; // no remapping by default
	return smat_subset_iterator_t<val_type> (*this, subset, subset_size, remmapping, smat_subset_iterator_t<val_type>::COLMAJOR);
} // }}}

template<typename val_type>
smat_t<val_type> smat_t<val_type>::row_subset(const std::vector<unsigned> &subset) const { // {{{
	return row_subset(&subset[0], (int)subset.size());
} // }}}

template<typename val_type>
smat_t<val_type> smat_t<val_type>::row_subset(const unsigned *subset, int subset_size) const { // {{{
	smat_subset_iterator_t<val_type> it(*this, subset, subset_size);
	smat_t<val_type> sub_smat;
	sub_smat.load_from_iterator(subset_size, cols, it.get_nnz(), &it);
	return sub_smat;
} // }}}

template<typename val_type>
val_type smat_t<val_type>::get_global_mean() const { // {{{
	val_type sum=0;
	for(size_t idx = 0; idx < nnz; idx++) sum += val[idx];
	return sum/(val_type)nnz;
} // }}}

template<typename val_type>
void smat_t<val_type>::remove_bias(val_type bias) { // {{{
	if(bias) {
		for(size_t idx = 0; idx < nnz; idx++) {
			val[idx] -= bias;
			val_t[idx] -= bias;
		}
	}
} // }}}

template<typename val_type>
val_type* smat_t<val_type>::Xv(const val_type *v, val_type *Xv) const { // {{{
	for(size_t i = 0; i < rows; ++i) {
		Xv[i] = 0;
		for(size_t idx = row_ptr[i]; idx < row_ptr[i+1]; ++idx)
			Xv[i] += val_t[idx] * v[col_idx[idx]];
	}
	return Xv;
} // }}}

template<typename val_type>
dvec_t<val_type>& smat_t<val_type>::Xv(const dvec_t<val_type>& v, dvec_t<val_type>& Xv) const { // {{{
	this->Xv(v.data(), Xv.data());
	return Xv;
} // }}}

template<typename val_type>
val_type* smat_t<val_type>::XTu(const val_type *u, val_type *XTu) const { // {{{
	for(size_t i = 0; i < cols; ++i) {
		XTu[i] = 0;
		for(size_t idx = col_ptr[i]; idx < col_ptr[i+1]; ++idx)
			XTu[i] += val[idx] * u[row_idx[idx]];
	}
	return XTu;
} // }}}

template<typename val_type>
dvec_t<val_type>& smat_t<val_type>::XTu(const dvec_t<val_type>& u, dvec_t<val_type>& XTu) const { // {{{
	this->XTu(u.data(), XTu.data());
	return XTu;
} // }}}

// Comparator for sorting rates into row/column comopression storage
template<typename val_type>
class SparseComp { // {{{
	public:
		const unsigned *row_idx;
		const unsigned *col_idx;
		SparseComp(const unsigned *row_idx_, const unsigned *col_idx_, bool isCSR=true) {
			row_idx = (isCSR)? row_idx_: col_idx_;
			col_idx = (isCSR)? col_idx_: row_idx_;
		}
		bool operator()(size_t x, size_t y) const {
			return  (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x]< col_idx[y]));
		}
}; // }}}

template<typename val_type>
void smat_t<val_type>::load_from_iterator(size_t _rows, size_t _cols, size_t _nnz, entry_iterator_t<val_type> *entry_it) { // {{{
	clear_space(); // clear any pre-allocated space in case of memory leak
	rows =_rows,cols=_cols,nnz=_nnz;
	allocate_space(rows,cols,nnz);

	// a trick here to utilize the space the have been allocated
	std::vector<size_t> perm(_nnz);
	unsigned *tmp_row_idx = col_idx;
	unsigned *tmp_col_idx = row_idx;
	val_type *tmp_val = val;
	for(size_t idx = 0; idx < _nnz; idx++){
		entry_t<val_type> rate = entry_it->next();
		row_ptr[rate.i+1]++;
		col_ptr[rate.j+1]++;
		tmp_row_idx[idx] = rate.i;
		tmp_col_idx[idx] = rate.j;
		tmp_val[idx] = rate.v;
		perm[idx] = idx;
	}
	// sort entries into row-majored ordering
	sort(perm.begin(), perm.end(), SparseComp<val_type>(tmp_row_idx, tmp_col_idx, true));
	// Generate CSR format
	for(size_t idx = 0; idx < _nnz; idx++) {
		val_t[idx] = tmp_val[perm[idx]];
		col_idx[idx] = tmp_col_idx[perm[idx]];
	}

	// Calculate nnz for each row and col
	max_row_nnz = max_col_nnz = 0;
	for(size_t r = 1; r <= rows; r++) {
		max_row_nnz = std::max(max_row_nnz, row_ptr[r]);
		row_ptr[r] += row_ptr[r-1];
	}
	for(size_t c = 1; c <= cols; c++) {
		max_col_nnz = std::max(max_col_nnz, col_ptr[c]);
		col_ptr[c] += col_ptr[c-1];
	}

	// Transpose CSR into CSC matrix
	for(size_t r = 0; r < rows; ++r){
		for(size_t idx = row_ptr[r]; idx < row_ptr[r+1]; idx++){
			size_t c = (size_t) col_idx[idx];
			row_idx[col_ptr[c]] = r;
			val[col_ptr[c]++] = val_t[idx];
		}
	}
	for(size_t c = cols; c > 0; --c) col_ptr[c] = col_ptr[c-1];
	col_ptr[0] = 0;
} // }}}

template<typename val_type>
void smat_t<val_type>::load(size_t _rows, size_t _cols, size_t _nnz, const char* filename, typename smat_t<val_type>::format_t fmt) { // {{{

	if(fmt == smat_t<val_type>::TXT) {
		file_iterator_t<val_type> entry_it(_nnz, filename);
		load_from_iterator(_rows, _cols, _nnz, &entry_it);
	} else if(fmt == smat_t<val_type>::PETSc) {
		load_from_PETSc(filename);
	} else {
		fprintf(stderr, "Error: filetype %d not supported\n", fmt);
		return ;
	}
} // }}}

template<typename val_type>
void smat_t<val_type>::save_PETSc_to_file(const char *filename) const { // {{{
	FILE *fp = fopen(filename, "wb");
	if(fp == NULL) {
		fprintf(stderr,"Error: can't open file %s\n", filename);
		exit(1);
	}
	save_PETSc_to_file(fp);
} // }}}

template<typename val_type>
void smat_t<val_type>::save_PETSc_to_file(FILE *fp) const { // {{{
	const int UNSIGNED_FILE = 1211216, LONG_FILE = 1015;
	int32_t int_buf[3] = {(int32_t)LONG_FILE, (int32_t)rows, (int32_t)cols};
	std::vector<int32_t> nnz_row(rows);
	for(size_t r = 0; r < rows; r++)
		nnz_row[r] = (int)nnz_of_row(r);

	fwrite(&int_buf[0], sizeof(int32_t), 3, fp);
	fwrite(&nnz, sizeof(size_t), 1, fp);
	fwrite(&nnz_row[0], sizeof(int32_t), rows, fp);
	fwrite(&col_idx[0], sizeof(unsigned), nnz, fp);

	// the following part == fwrite(val_t, sizeof(double), nnz, fp);
	const size_t chunksize = 1024;
	double buf[chunksize];
	size_t idx = 0;
	while(idx + chunksize < nnz) {
		for(size_t i = 0; i < chunksize; i++)
			buf[i] = (double) val_t[idx+i];
		fwrite(&buf[0], sizeof(double), chunksize, fp);
		idx += chunksize;
	}
	size_t remaining = nnz - idx;
	for(size_t i = 0; i < remaining; i++)
		buf[i] = (double) val_t[idx+i];
	fwrite(&buf[0], sizeof(double), remaining, fp);
} // }}}

template<typename val_type>
void smat_t<val_type>::load_from_PETSc(const char *filename) { // {{{
	FILE *fp = fopen(filename, "rb");
	if(fp == NULL) {
		fprintf(stderr, "Error: can't read the file (%s)!!\n", filename);
		return;
	}
	load_from_PETSc(fp, filename);
	fclose(fp);
} // }}}

template<typename val_type>
void smat_t<val_type>::load_from_PETSc(FILE *fp, const char *filename) { // {{{
	clear_space(); // clear any pre-allocated space in case of memory leak
	const int UNSIGNED_FILE = 1211216, LONG_FILE = 1015;
	int32_t int_buf[3];
	size_t headersize = 0;
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
		fprintf(stderr, "Error: wrong PETSc format in %s.\n", filename);
	}
	allocate_space(rows,cols,nnz);
	// load CSR from the binary PETSc format
	{ // {{{
		// read row_ptr
		std::vector<int32_t> nnz_row(rows);
		headersize += sizeof(int32_t)*fread(&nnz_row[0], sizeof(int32_t), rows, fp);
		row_ptr[0] = 0;
		for(size_t r = 1; r <= rows; r++)
			row_ptr[r] = row_ptr[r-1] + nnz_row[r-1];
		// read col_idx
		headersize += sizeof(int)*fread(&col_idx[0], sizeof(unsigned), nnz, fp);

		// read val_t
		const size_t chunksize = 1024;
		double buf[chunksize];
		size_t idx = 0;
		while(idx + chunksize < nnz) {
			headersize += sizeof(double)*fread(&buf[0], sizeof(double), chunksize, fp);
			for(size_t i = 0; i < chunksize; i++)
				val_t[idx+i] = (val_type) buf[i];
			idx += chunksize;
		}
		size_t remaining = nnz - idx;
		headersize += sizeof(double)*fread(&buf[0], sizeof(double), remaining, fp);
		for(size_t i = 0; i < remaining; i++)
			val_t[idx+i] = (val_type) buf[i];
	} // }}}

	csr_to_csc();
	update_max_nnz();
} // }}}

template<typename val_type>
void smat_t<val_type>::csr_to_csc() { // {{{
	memset(col_ptr, 0, sizeof(size_t)*(cols+1));
	for(size_t idx = 0; idx < nnz; idx++)
		col_ptr[col_idx[idx]+1]++;
	for(size_t c = 1; c <= cols; c++)
		col_ptr[c] += col_ptr[c-1];
	for(size_t r = 0; r < rows; r++) {
		for(size_t idx = row_ptr[r]; idx != row_ptr[r+1]; idx++) {
			size_t c = (size_t) col_idx[idx];
			row_idx[col_ptr[c]] = r;
			val[col_ptr[c]++] = val_t[idx];
		}
	}
	for(size_t c = cols; c > 0; c--)
		col_ptr[c] = col_ptr[c-1];
	col_ptr[0] = 0;
} // }}}

template<typename val_type>
void smat_t<val_type>::csc_to_csr() { // {{{
	memset(row_ptr, 0, sizeof(size_t)*(rows+1));
	for(size_t idx = 0; idx < nnz; idx++)
		row_ptr[row_idx[idx]+1]++;
	for(size_t r = 1; r <= rows; r++)
		row_ptr[r] += row_ptr[r-1];
	for(size_t c = 0; c < cols; c++) {
		for(size_t idx = col_ptr[c]; idx != col_ptr[c+1]; idx++) {
			size_t r = (size_t) row_idx[idx];
			col_idx[row_ptr[r]] = c;
			val_t[row_ptr[r]++] = val[idx];
		}
	}
	for(size_t r = rows; r > 0; r--)
		row_ptr[r] = row_ptr[r-1];
	row_ptr[0] = 0;
} // }}}

template<typename val_type>
void smat_t<val_type>::update_max_nnz() { // {{{
	max_row_nnz = max_col_nnz = 0;
	for(size_t c = 0; c < cols; c++) max_col_nnz = std::max(max_col_nnz, nnz_of_col(c));
	for(size_t r = 0; r < rows; r++) max_row_nnz = std::max(max_row_nnz, nnz_of_row(r));
} // }}}

template<typename val_type>
file_iterator_t<val_type>::file_iterator_t(size_t nnz_, const char* filename, size_t start_pos) { // {{{
	nnz = nnz_;
	fp = fopen(filename,"rb");
	if(fp == NULL) {
		fprintf(stderr, "Error: cannot read the file (%s)!!\n", filename);
		return;
	}
	fseek(fp, start_pos, SEEK_SET);
} // }}}

template<typename val_type>
entry_t<val_type> file_iterator_t<val_type>::next() { // {{{
	const int base10 = 10;
	if(nnz > 0) {
		--nnz;
		if(fgets(&line[0], MAXLINE, fp)==NULL)
			fprintf(stderr, "Error: reading error !!\n");
		char *head_ptr = &line[0];
		size_t i = strtol(head_ptr, &head_ptr, base10);
		size_t j = strtol(head_ptr, &head_ptr, base10);
		double v = strtod(head_ptr, &head_ptr);
		return entry_t<val_type>(i-1, j-1, (val_type)v);
	} else {
		fprintf(stderr, "Error: no more entry to iterate !!\n");
		return entry_t<val_type>(0,0,0);
	}
} // }}}

template<typename val_type>
smat_iterator_t<val_type>::smat_iterator_t(const smat_t<val_type>& M, major_t major) { // {{{
	nnz = M.nnz;
	col_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
	row_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
	val_t = (major == ROWMAJOR)? M.val_t: M.val;
	rows = (major==ROWMAJOR)? M.rows: M.cols;
	cols = (major==ROWMAJOR)? M.cols: M.rows;
	cur_idx = cur_row = 0;
} // }}}

template<typename val_type>
entry_t<val_type> smat_iterator_t<val_type>::next() { // {{{
	while (cur_idx >= row_ptr[cur_row+1])
		cur_row++;
	if (nnz > 0)
		nnz--;
	else
		fprintf(stderr,"Error: no more entry to iterate !!\n");
	entry_t<val_type> ret(cur_row, col_idx[cur_idx], val_t[cur_idx]);
	cur_idx++;
	return ret;
} // }}}

template<typename val_type>
smat_subset_iterator_t<val_type>::smat_subset_iterator_t(const smat_t<val_type>& M, const unsigned *subset, size_t size, bool remapping_, major_t major_) { // {{{
	major = major_; remapping = remapping_;
	col_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
	row_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
	val_t = (major == ROWMAJOR)? M.val_t: M.val;
	rows = (major==ROWMAJOR)? (remapping?size:M.rows): (remapping?size:M.cols);
	cols = (major==ROWMAJOR)? M.cols: M.rows;
	this->subset.resize(size);
	nnz = 0;
	for(size_t i = 0; i < size; i++) {
		unsigned idx = subset[i];
		this->subset[i] = idx;
		nnz += (major == ROWMAJOR)? M.nnz_of_row(idx): M.nnz_of_col(idx);
	}
	sort(this->subset.begin(), this->subset.end());
	cur_row = 0;
	cur_idx = row_ptr[this->subset[cur_row]];
} // }}}

template<typename val_type>
entry_t<val_type> smat_subset_iterator_t<val_type>::next() { // {{{
	while (cur_idx >= row_ptr[subset[cur_row]+1]) {
		cur_row++;
		cur_idx = row_ptr[subset[cur_row]];
	}
	if (nnz > 0)
		nnz--;
	else
		fprintf(stderr,"Error: no more entry to iterate !!\n");
	//entry_t<val_type> ret(cur_row, col_idx[cur_idx], val_t[cur_idx]);
	entry_t<val_type> ret_rowwise(remapping?cur_row:subset[cur_row], col_idx[cur_idx], val_t[cur_idx]);
	entry_t<val_type> ret_colwise(col_idx[cur_idx], remapping?cur_row:subset[cur_row], val_t[cur_idx]);
	//printf("%d %d\n", cur_row, col_idx[cur_idx]);
	cur_idx++;
	//return ret;
	return major==ROWMAJOR? ret_rowwise: ret_colwise;
} // }}}


/*
   H = X*W
   X is an m*n
   W is an n*k, row-majored array
   H is an m*k, row-majored array
   */
template<typename val_type>
void smat_x_dmat(const smat_t<val_type> &X, const val_type* W, const size_t k, val_type *H) { // {{{
	size_t m = X.rows;
#pragma omp parallel for schedule(dynamic,50) shared(X,W,H)
	for(size_t i = 0; i < m; i++) {
		val_type *Hi = &H[k*i];
		memset(Hi,0,sizeof(val_type)*k);
		for(size_t idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
			const val_type Xij = X.val_t[idx];
			const val_type *Wj = &W[X.col_idx[idx]*k];
			for(unsigned t = 0; t < k; t++)
				Hi[t] += Xij*Wj[t];
		}
	}
} // }}}
template<typename val_type>
void smat_x_dmat(const smat_t<val_type> &X, const dmat_t<val_type> &W, dmat_t<val_type> &H) { // {{{
	assert(W.cols == H.cols && X.cols == W.rows && X.rows == H.rows);
	assert(W.is_rowmajor() && H.is_rowmajor());
	smat_x_dmat(1.0, X, W, 0.0, H, H);
	//smat_x_dmat(X, W.data(), W.cols, H.data());
} // }}}


/*
   H = a*X*W + b H0
   X is an m*n
   W is an n*k, row-majored array
   H is an m*k, row-majored array
   */

template<typename val_type, typename T2, typename T3>
void smat_x_dmat(T2 a, const smat_t<val_type> &X, const val_type *W, const size_t k, T3 b, const val_type *H0, val_type *H) { // {{{
	size_t m = X.rows;
	val_type aa = (val_type) a;
	val_type bb = (val_type) b;
	if(a == T2(0)) {
		if(bb == (val_type)0.0){
			memset(H, 0, sizeof(val_type)*m*k);
			return ;
		} else {
			if(H!=H0) {
				do_copy(H0, H, m*k);
				//memcpy(H, H0, sizeof(val_type)*m*k);
			}
			do_scale(bb, H, m*k);
		}
		return;
	}
#pragma omp parallel for schedule(dynamic,64) shared(X, W, H, H0, aa,bb)
	for(size_t i = 0; i < m; i++) {
		val_type *Hi = &H[k*i];
		if(bb == (val_type)0.0)
			memset(Hi, 0, sizeof(val_type)*k);
		else {
			if(Hi!=&H0[k*i])
				do_copy(&H0[k*i], Hi, k);
			do_scale(bb, Hi, k);
		}
		for(size_t idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
			const val_type Xij = X.val_t[idx];
			const val_type *Wj = &W[X.col_idx[idx]*k];
			for(size_t t = 0; t < k; t++)
				Hi[t] += aa*Xij*Wj[t];
		}
	}

}// }}}

template<typename val_type, typename T2>
void smat_x_dmat(T2 a, const smat_t<val_type> &X, const val_type* W, const size_t k, const val_type *H0, val_type *H) { // {{{
	smat_x_dmat(a, X, W, k, 1.0, H0, H);
} // }}}

template<typename val_type, typename T2, typename T3>
void smat_x_dmat(T2 a, const smat_t<val_type> &X, const dmat_t<val_type> &W, T3 b, const dmat_t<val_type> &H0, dmat_t<val_type> &H) { // {{{
	assert(W.cols == H0.cols && W.cols == H.cols && X.cols == W.rows && X.rows == H0.rows && X.rows == H.rows);
	if(W.is_rowmajor()) {
		if(H.is_rowmajor()) {
			if(H0.is_rowmajor()){
				smat_x_dmat(a, X, W.data(), W.cols, b, H0.data(), H.data());
			} else {
				H.assign(b, H0);
				smat_x_dmat(a, X, W.data(), W.cols, 1.0, H.data(), H.data());
			}
		} else { // H is col_major
			H.assign(b, H0);
			// H += aXW
#pragma omp parallel for schedule(dynamic, 64)  shared(X, W, H)
			for(size_t i = 0; i < X.rows; i++) {
				for(size_t idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++){
					size_t j = X.col_idx[idx];
					const val_type &Xij = X.val_t[idx];
					for(size_t t = 0; t < W.cols; t++)
						H.at(i,t) += a*Xij*W.at(j,t);
				}
			}
		}
	} else { // W.is_colmajor
		H.assign(b, H0);
		if(H.is_colmajor()) {
#pragma omp parallel for schedule(static)
			for(size_t j = 0; j < W.cols; j++)
				X.Xv(W[j], H[j]);
		} else { // H.is row_major
			// H += aXW
#pragma omp parallel for schedule(dynamic, 64)  shared(X, W, H)
			for(size_t i = 0; i < X.rows; i++) {
				for(size_t idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++){
					size_t j = X.col_idx[idx];
					const val_type &Xij = X.val_t[idx];
					for(size_t t = 0; t < W.cols; t++)
						H.at(i,t) += a*Xij*W.at(j,t);
				}
			}
		}
	}
}// }}}

template<typename val_type, typename T2>
void smat_x_dmat(T2 a, const smat_t<val_type> &X, const dmat_t<val_type> &W, const dmat_t<val_type> &H0, dmat_t<val_type> &H) { // {{{
	smat_x_dmat(a, X, W, 1.0, H0, H);
} // }}}

/*
 * H = a*XW + b H0
 * X is an m*n gmat
 * W is an n*k dmat
 * H is m*k dmat
 */
template<typename val_type, typename T2, typename T3>
void gmat_x_dmat(T2 a, const gmat_t<val_type>& X, const dmat_t<val_type>& W, T3 b, const dmat_t<val_type>& H0, dmat_t<val_type>& H) { // {{{
	if(X.is_sparse())
		smat_x_dmat(a, X.get_sparse(), W, b, H0, H);
	else if(X.is_dense())
		dmat_x_dmat(a, X.get_dense(), W, b, H0, H);
	else if(X.is_identity()) {
		H.assign(b, H0);
		do_axpy(a, W, H);
	}
} // }}}

/*
 * H = XW
 *
 */
template<typename val_type>
void gmat_x_dmat(const gmat_t<val_type>& X, const dmat_t<val_type>& W, dmat_t<val_type>& H) { // {{{
	if(X.is_sparse())
		smat_x_dmat(X.get_sparse(), W, H);
	else if(X.is_dense())
		dmat_x_dmat(X.get_dense(), W, H);
	else if(X.is_identity())
		H.assign(W);
} // }}}

/*
	trace(W^T X H)
	X is an m*n, sparse matrix
	W is an m*k, row-majored array
	H is an n*k, row-major
*/
template<typename val_type>
val_type trace_dmat_T_smat_dmat(const val_type *W, const smat_t<val_type> &X, const val_type *H, const size_t k) { // {{{
	size_t m = X.rows;
	double ret = 0;
#pragma omp parallel for schedule(dynamic,50) shared(X,H,W) reduction(+:ret)
	for(size_t i = 0; i < m; i++) {
		const val_type *Wi = &W[k*i];
		for(long idx = X.row_ptr[i]; idx < X.row_ptr[i+1]; idx++) {
			const val_type *Hj = &H[X.col_idx[idx]*k];
			double tmp=0;
			for(size_t t = 0; t < k; t++)
				tmp += Wi[t]*Hj[t];
			ret += X.val_t[idx]*tmp;
		}
	}
	return (val_type)ret;
} // }}}
template<typename val_type>
val_type trace_dmat_T_smat_dmat(const dmat_t<val_type> &W, const smat_t<val_type> &X, const dmat_t<val_type> &H) { // {{{
	assert(W.cols == H.cols && W.rows == X.rows && H.rows == X.cols);
	if(W.is_colmajor() && H.is_colmajor()) {
		double ret = 0;
#pragma omp parallel for schedule(static) reduction(+:ret)
		for(size_t t = 0; t < W.cols; t++) {
			const dvec_t<val_type> &u = W[t];
			const dvec_t<val_type> &v = H[t];
			double local_sum = 0;
			for(size_t i = 0; i < X.rows; i++) {
				for(size_t idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++)
					local_sum += X.val_t[idx]*u[i]*v[X.col_idx[idx]];
			}
			ret += local_sum;
		}
		return ret;
	} else {
		double ret= 0;
#pragma omp parallel for schedule(dynamic,64) reduction(+:ret)
		for(size_t i = 0; i < X.rows; i++) {
			double  local_sum = 0;
			for(size_t idx = X.row_ptr[i]; idx != X.row_ptr[i+1]; idx++) {
				size_t j = X.col_idx[idx];
				double sum = 0;
				for(size_t t = 0; t < W.cols; t++)
					sum += W.at(i,t)*H.at(j,t);
				local_sum += sum * X.val_t[idx];
			}
			ret += local_sum;
		}
		return ret;
	}
} // }}}

/*
	trace(W^T diag(D) H)
	D is an m*1 vector
	W is an m*k, row-majored array
	H is an m*k, row-major array
 */
template<typename val_type>
val_type trace_dmat_T_diag_dmat(const val_type *W, const val_type *D, const val_type *H, const size_t m, const size_t k) { // {{{
	val_type *w = const_cast<val_type*>(W);
	val_type *h = const_cast<val_type*>(H);
	val_type *d = const_cast<val_type*>(D);
	double ret = 0.0;
#pragma omp parallel for schedule(static) shared(w,h,d) reduction(+:ret)
	for(size_t i = 0; i < m; i++) {
		val_type *wi = &w[i*k], *hi = &h[i*k];
		ret += do_dot_product(wi, wi, k) * d[i];
	}
	return (val_type)ret;
} // }}}
template<typename val_type>
val_type trace_dmat_T_diag_dmat(const dmat_t<val_type> &W, const dvec_t<val_type> &D, const dmat_t<val_type> &H) { // {{{
	assert(W.rows == H.rows && W.rows == D.len && W.cols == H.cols);
	assert(W.is_rowmajor() && H.is_rowmajor());
	return trace_dmat_T_diag_dmat(W.data(),D.data(),H.data(),W.rows,W.cols);
} // }}}
template<typename val_type>
val_type trace_dmat_T_diag_dmat(const dmat_t<val_type> &W, const dmat_t<val_type> &D, const dmat_t<val_type> &H) { // {{{
	return trace_dmat_T_diag_dmat(W, dvec_t<val_type>(D.get_view()), H);
} // }}}


//------------------ Implementation of zip_it -----------------------
// helpler functions and classes for zip_it
template<class T1, class T2>
struct zip_body { // {{{
	T1 x; T2 y;
	zip_body(const zip_ref<T1,T2>& other): x(*other.x), y(*other.y){}
	bool operator<(const zip_body &other) const {return x < other.x;}
	bool operator>(zip_body &other) const {return x > other.x;}
	bool operator==(zip_body &other) const {return x == other.x;}
	bool operator!=(zip_body &other) const {return x != other.x;}
}; // }}}

template<class T1, class T2>
struct zip_ref { // {{{
	T1 *x; T2 *y;
	zip_ref(T1 &x, T2 &y): x(&x), y(&y){}
	zip_ref(zip_body<T1,T2>& other): x(&other.x), y(&other.y){}
	bool operator<(zip_ref other) const {return *x < *other.x;}
	bool operator>(zip_ref other) const {return *x > *other.x;}
	bool operator==(zip_ref other) const {return *x == *other.x;}
	bool operator!=(zip_ref other) const {return *x != *other.x;}
	zip_ref& operator=(zip_ref& other) {
		*x = *other.x; *y = *other.y;
		return *(this);
	}
	zip_ref& operator=(zip_body<T1,T2> other) {
		*x = other.x; *y = other.y;
		return *(this);
	}
}; // }}}

template<class T1, class T2>
void swap(zip_ref<T1,T2> a, zip_ref<T1,T2> b) { // {{{
	std::swap(*(a.x),*(b.x));
	std::swap(*(a.y),*(b.y));
} // }}}

template<class IterT1, class IterT2>
struct zip_it { // {{{
	typedef std::random_access_iterator_tag iterator_category;
	typedef typename std::iterator_traits<IterT1>::value_type T1;
	typedef typename std::iterator_traits<IterT2>::value_type T2;
	typedef zip_body<T1,T2> value_type;
	typedef zip_ref<T1,T2> reference;
	typedef zip_body<T1,T2>* pointer;
	typedef ptrdiff_t difference_type;
	IterT1 x;
	IterT2 y;
	zip_it(IterT1 x, IterT2 y): x(x), y(y){}
	reference operator*() {return reference(*x, *y);}
	reference operator[](const difference_type n) const {return reference(x[n],y[n]);}
	zip_it& operator++() {++x; ++y; return *this;} // prefix ++
	zip_it& operator--() {--x; --y; return *this;} // prefix --
	zip_it operator++(int) {return zip_it(x++,y++);} // sufix ++
	zip_it operator--(int) {return zip_it(x--,y--);} // sufix --
	zip_it operator+(const difference_type n) {return zip_it(x+n,y+n);}
	zip_it operator-(const difference_type n) {return zip_it(x-n,y-n);}
	zip_it& operator+=(const difference_type n) {x+=n; y+=n; return *this;}
	zip_it& operator-=(const difference_type n) {x-=n; y-=n; return *this;}
	bool operator<(const zip_it& other) {return x<other.x;}
	bool operator>(const zip_it& other) {return x>other.x;}
	bool operator==(const zip_it& other) {return x==other.x;}
	bool operator!=(const zip_it& other) {return x!=other.x;}
	difference_type operator-(const zip_it& other) {return x-other.x;}
}; // }}}

template<class IterT1, class IterT2>
zip_it<IterT1, IterT2> zip_iter(IterT1 x, IterT2 y) { // {{{
	return zip_it<IterT1,IterT2>(x,y);
} // }}}

//}; // end of namespace rofu

#undef gmat_t
#undef eye_t
#undef smat_t
#undef dmat_t
#undef dvec_t
#endif // SPARSE_MATRIX_H
