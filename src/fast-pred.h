#ifndef FAST_PRED_H
#define FAST_PRED_H

#include "pmf.h"
#include <queue>
#include <ctime>

struct increasing_comp_t { // {{{
	val_type *pred_val;
	increasing_comp_t(val_type *_val): pred_val(_val) {}
	bool operator()(size_t i, size_t j) const {return pred_val[i] < pred_val[j];}
}; // }}}

class topk_iterator_t : public entry_iterator_t <val_type> { // {{{
	private:
		size_t rows, cols, topk, cur_row, cur_k;
		const unsigned *pred_topk;
		int idx_base;
		size_t nnz;
	public :
		topk_iterator_t(size_t rows, size_t cols, size_t topk, const unsigned *pred_topk, int idx_base=0): rows(rows), cols(cols), topk(topk), cur_row(0), cur_k(0), pred_topk(pred_topk), idx_base(idx_base) { // {{{
			nnz = 0;
			bool search_first = true;
			for(size_t row = 0; row < rows; row++)
				for(int t = 0; t < topk; t++)
					if((pred_topk[row*topk+t]-idx_base) < cols) {
						nnz++;
						if(search_first) {
							cur_row = row;
							cur_k = t;
							search_first = false;
						}
					}
			printf("cur_row %lu, cur_k %lu nnz %ld\n", cur_row, cur_k, nnz);
		} // }}}
		entry_t<val_type> next() { // {{{
			entry_t<val_type> entry(cur_row, (unsigned)(pred_topk[cur_row*topk+cur_k]-idx_base), (val_type)(topk-cur_k));
			do {
				cur_k += 1;
				if(cur_k == topk) {
					cur_row += 1;
					cur_k = 0;
				}
				if(cur_row == rows) break;
			} while((pred_topk[cur_row*topk+cur_k]-idx_base) >= cols);
			return entry;
		} // }}}
		size_t get_nnz() const {return nnz;}
}; // }}}

// Y = W*H, instead of W*H' =>  H.cols = nr_candidate, W.rows = nr_query
struct test_case_t {
	dmat_t H; // col-major   // data base items
	dmat_t W; // row-major   // query items
	smat_t true_rank;
	size_t topk;

	size_t nr_query() const {return W.rows;}
	size_t nr_candidate() const {return H.cols;}
	const dvec_t &query(size_t qid) const {return W[qid];}
	void init(const dmat_t &W_, const dmat_t &H_, size_t topk_=20) { // {{{
		topk = topk_;
		W = W_.get_view(); H = H_.get_view();
		W.to_rowmajor(); H.to_colmajor();
		auto TT=std::vector<unsigned>(W.rows*topk);
		pmf_ranker_t ranker;
		ranker.predict(W, H.transpose(), topk, TT.data());
		topk_iterator_t it(W.rows, H.cols, topk, TT.data());
		true_rank.load_from_iterator(W.rows, H.cols, it.get_nnz(), &it);
	} // }}}
	void save_binary_to_file(FILE *fp) { // {{{
		fwrite(&topk, sizeof(size_t), 1, fp);
		H.save_binary_to_file(fp);
		W.save_binary_to_file(fp);
		true_rank.save_PETSc_to_file(fp);
	}
	void save_binary_to_file(const char *filenmae) {
		FILE *fp = fopen(filenmae, "wb");
		save_binary_to_file(fp);
		fclose(fp);
	} // }}}
	void load_from_binary(FILE *fp) { // {{{
		int tmp = fread(&topk, sizeof(size_t), 1, fp);
		H.load_from_binary(fp, COLMAJOR);
		W.load_from_binary(fp, ROWMAJOR);
		true_rank.load_from_PETSc(fp);
	}
	void load_from_binary(const char *filename) {
		FILE *fp = fopen(filename, "rb");
		load_from_binary(fp);
		fclose(fp);
	} // }}}
};

template<typename T=double>
struct htree_t{ // {{{
	size_t size;     // real size of valid elements
	size_t elements; // 2^ceil(log2(size)) capacity
	std::vector<T> val;
	T *true_val;

	T& operator[](size_t idx) { assert(idx < elements); return true_val[idx]; }
	const T& operator[] (size_t idx) const { assert(idx < elements); return true_val[idx]; }
	void init_dense() { // {{{
		/*
		for(size_t pos = (elements+size)>>1; pos > 0; --pos)
			val[pos] = val[pos<<1] + val[(pos<<1)+1];
			*/
		for(size_t pos = elements-1; pos > 0; --pos)
			val[pos] = val[pos<<1] + val[(pos<<1)+1];
	} // }}}
	void update_parent(size_t idx, T delta) { // {{{
		idx = (idx+elements)>>1;
		while(idx) {
			val[idx] += delta;
			idx >>= 1;
		}
	} // }}}
	void set_value(size_t idx, T value) { // {{{
		value -= val[idx+=elements]; // delta update
		while(idx) {
			val[idx] += value;
			idx >>= 1;
		}
	} // }}}
	// urnd: uniformly random number between [0,1]
	size_t log_sample(double urnd) { // {{{
		//urnd *= val[1];
		size_t pos = 1;
		while(pos < elements) {
		//while(pos < size) {
			pos <<= 1;
			if(urnd > val[pos])
				urnd -= val[pos++];
			/*
			double tmp = urnd - val[pos];
			if(tmp >= 0) {
				urnd = tmp;
				pos++;
			}
			*/
			/*
			if(urnd < val[pos*2])
				pos = pos*2;
			else {
				urnd -= val[pos*2];
				pos = pos*2+1;
			}
			*/
		}
		return pos-elements;
	} // }}}
	size_t linear_sample(double urnd) { // {{{
		//urnd = urnd*val[1];
		size_t pos = elements;
		while(urnd > 0)
			urnd -= val[pos++];
		if(pos >= elements+size) pos = elements+size-1;
		return pos-elements;
	} // }}}
	double total_sum() { return val[1]; }
	double left_cumsum(size_t idx) { // {{{
		if(idx == elements) return val[1];
		size_t pos = elements+idx+1;
		double sum = 0;
		while(pos>1) {
			if(pos & 1)
				sum += val[pos^1];
			pos >>= 1;
		}
		return sum;
	} // }}}
	double right_cumsum(size_t idx) {return val[1] - left_cumsum(idx-1);}
	htree_t(size_t size=0) { resize(size); }
	htree_t(const htree_t& other) {
		size = other.size;
		elements = other.elements;
		val = other.val;
		true_val = &val[elements];
	}
	htree_t& operator=(const htree_t& other) {
		size = other.size;
		elements = other.elements;
		val = other.val;
		true_val = &val[elements];
	}

	void resize(size_t size_) { // {{{
		size = size_;
		if(size == 0) {
			val.clear(); elements = 0; true_val = NULL;
			return;
		} else {
			elements = 1;
			while(elements < size) elements <<=1;
			val.clear(); val.resize(2*elements, 0);
			true_val = &val[elements];
		}
		//Q.reserve(elements);
	} //}}}
	void clear() { for(auto &v: val) v = 0; }
}; // }}}

struct arg_max { // {{{
	double value;
	size_t idx;
	arg_max(size_t idx=0, double value=-1e300): value(value), idx(idx){}
	bool operator<(const arg_max &other) const {return value < other.value;}

	static arg_max default_value() {return arg_max(0, -1e300);}
	static const arg_max& op(const arg_max& a, const arg_max& b) { return a.value < b.value ? b: a; }
}; // }}}

// descending comparator for arg_sort
template<typename T>
struct comparator { // {{{
	const T* value;
	comparator(const T* value=NULL): value(value){}
	bool operator()(size_t a, size_t b) const { return value[a] > value[b]; }
}; // }}}

// An implementation of Algorithm 3: selection tree described in TAOCP by Knuth [8]
template<typename T>
struct Ftree_t { // {{{
	size_t len ;     // real length of valid elements
	size_t elements; // 2^ceil(log2(len)) capacity
	std::vector<T> val;
	T *true_val;

	Ftree_t(size_t len_= 0) {resize(len_);}
	Ftree_t(const Ftree_t &other) { // {{{
		len = other.len;
		elements = other.elements;
		val = other.val;
		true_val = &val[elements];
	} // }}}
	Ftree_t& operator=(const Ftree_t &other) { // {{{
		len = other.len;
		elements = other.elements;
		val = other.val;
		true_val = &val[elements];
	} // }}}k
	size_t size() const {return len;}
	void resize(size_t len_) { // {{{
		len = len_;
		if(len == 0){
			val.clear(); elements = 0; true_val = NULL;
			return;
		} else {
			elements = 1;
			while (elements<len) elements <<= 1;
			val.clear(); val.resize(2*elements, T::default_value());
			true_val = &val[elements];
		}
	} // }}}
	void clear() {resize(0);}

	T &operator[](size_t idx) {return true_val[idx];}
	const T &operator[](size_t idx) const {return true_val[idx];}

	void init_dense() { // {{{
		for(size_t pos = elements-1; pos > 0; --pos)
			val[pos] = T::op(val[pos<<1], val[(pos<<1)+1]);
	} // }}}
	void set_value(size_t idx, const T& v) { // {{{
		val[idx+=elements] = v;
		while (idx>1) {
			val[idx>>1] = T::op(val[idx],val[idx^1]);
			idx >>=1;
		}
	} // }}}
	void print() { // {{{
		std::cout << "len " << len << "; elements " << elements << std::endl;
		size_t idx = 1;
		while (idx <= elements) {
			for(size_t pos = idx; pos < (idx << 1); pos++)
				std::cout << "(" << val[pos].idx<<"," <<val[pos].value<< ") ";
			std::cout << std::endl;
			idx <<= 1;
		}
	} // }}}
}; // }}}

struct greedy_mips_t { // {{{
	typedef std::vector<size_t> candidate_t;

	typedef std::vector<arg_max> pool_t;
	std::vector<pool_t> pools;
	struct pool_view_t { // {{{
		const pool_t* p;
		bool increasing;
		size_t idx;
		pool_view_t(): p(NULL), increasing(true), idx(0){}
		pool_view_t(const pool_t& p, bool increasing =true): p(&p), increasing(increasing), idx(0) {}
		const arg_max& head() const {return (*this)[idx];}
		bool empty() {return idx == p->size();}
		void pop() {idx++;}
		const arg_max& operator[](size_t idx) const {
			if(increasing) return (*p)[idx];
			else return (*p)[p->size()-idx-1];
		}
	}; // }}}

	const dmat_t M;
	std::vector<size_t> head_of_pools;
	std::vector<size_t> g_cnt;
	std::vector<double> g_val;

	greedy_mips_t(const dmat_t &M): M(M.get_view()) { // {{{
		pools.resize(M.rows, pool_t(M.cols));
		g_cnt.resize(M.cols);
		g_val.resize(M.cols);

		for(size_t i = 0; i < M.rows; i++) {
			auto &Q = pools[i];
			for(size_t j = 0; j < M.cols; j++)
				Q[j] = arg_max(j, M.at(i,j));
			std::sort(Q.begin(), Q.end()); // increasing order
		}
	} // }}}

	double search(const dvec_t& w, size_t budget, candidate_t& candidate) { // {{{
		candidate.clear();
		if(w.size() != pools.size()) {
			fprintf(stderr,"query with wrong size\n");
			return 0;
		}
		auto tic = std::clock();
		std::vector<pool_view_t> pool_views(w.size());
		Ftree_t<arg_max> Ft(w.size());
		for(size_t t = 0; t < w.size(); t++) {
			pool_views[t] = pool_view_t(pools[t], w[t]<0);
			Ft[t] = arg_max(t, pool_views[t].head().value * w[t]);
		}
		Ft.init_dense();
		budget = std::min(budget, pools.size()*pools[0].size()-1);
		for(size_t b = 0; b < budget; b++) {
			auto t = Ft.val[1].idx;     // pool id
			auto v = Ft.val[1].value;   // entry value
			auto& cur_pool= pool_views[t];
			auto i = cur_pool.head().idx; // candidate id
			if(g_cnt[i] == 0)
				candidate.push_back(i);
			g_cnt[i] += 1;
			g_val[i] += v;

			cur_pool.pop();
			if(cur_pool.empty())
				Ft.set_value(t, arg_max());
			else
				Ft.set_value(t, arg_max(t,cur_pool.head().value * w[t]));
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
	double search_cnt(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		std::sort(candidate.begin(), candidate.end(), comparator<size_t>(g_cnt.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
	double search_val(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);

	} // }}}
	double search_true(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		for(auto &i : candidate) {
			double sum = 0;
			for(size_t t = 0; t < w.size(); t++)
				sum += w[t] * M.at(t,i);
			g_val[i] = sum;
		}
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
}; // }}}

// Greedy MIPS with Selection Tree: An Implementation of Algorithm 3 with Selection Tree (Algorithm 4)
struct improved_greedy_mips_t { // {{{
	typedef std::vector<size_t> candidate_t;

	typedef std::vector<arg_max> pool_t;
	std::vector<pool_t> pools;
	// An Implementation of Algorithm 1 (CondIter)
	struct pool_view_t { // {{{
		const pool_t* p;
		bool increasing;
		size_t idx;
		pool_view_t(): p(NULL), increasing(true), idx(0){}
		pool_view_t(const pool_t& p, bool increasing=true): p(&p), increasing(increasing), idx(0) {}
		const arg_max& head() const {return (*this)[idx];}
		bool empty() {return idx == p->size();}
		void pop() {idx++;}
		const arg_max& operator[](size_t idx) const {
			if(increasing) return (*p)[idx];
			else return (*p)[p->size()-idx-1];
		}
	}; // }}}

	const dmat_t M;
	std::vector<size_t> head_of_pools;
	std::vector<size_t> g_cnt;
	std::vector<double> g_val;

	improved_greedy_mips_t(const dmat_t &M): M(M.get_view()) { // {{{
		pools.resize(M.rows, pool_t(M.cols));
		g_cnt.resize(M.cols);
		g_val.resize(M.cols);

		for(size_t i = 0; i < M.rows; i++) {
			auto &Q = pools[i];
			for(size_t j = 0; j < M.cols; j++)
				Q[j] = arg_max(j, M.at(i,j));
			std::sort(Q.begin(), Q.end()); // increasing order
		}
	} // }}}

	double search(const dvec_t& w, size_t budget, candidate_t& candidate) { // {{{
		candidate.clear();
		if(w.size() != pools.size()) {
			fprintf(stderr,"query with wrong size\n");
			return 0;
		}
		auto tic = std::clock();
		// Algorithm 2
		std::vector<pool_view_t> pool_views(w.size());
		Ftree_t<arg_max> Ft(w.size());
		for(size_t t = 0; t < w.size(); t++) {
			pool_views[t] = pool_view_t(pools[t], w[t]<0);
			Ft[t] = arg_max(t, pool_views[t].head().value * w[t]);
		}
		Ft.init_dense();

		budget = std::min(budget, pools.size()*pools[0].size()-1);
		for(size_t b = 0; b < budget; b++) {
			auto t = Ft.val[1].idx;     // pool id
			auto v = Ft.val[1].value;   // entry value
			auto& cur_pool= pool_views[t];
			auto i = cur_pool.head().idx; // candidate id
			if(g_cnt[i] == 0)
				candidate.push_back(i);
			g_cnt[i] += 1;
			g_val[i] += v;

			cur_pool.pop();

			while(1) {
				if(cur_pool.empty()) {
					Ft.set_value(t, arg_max());
					break;
				}
				const auto& iter  = cur_pool.head();
				if(g_cnt[iter.idx] == 0) {
					Ft.set_value(t, arg_max(t, iter.value*w[t]));
					break ;
				}
				cur_pool.pop();
				b+=1;
			}
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
	double search_cnt(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		std::sort(candidate.begin(), candidate.end(), comparator<size_t>(g_cnt.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
	double search_val(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);

	} // }}}
	double search_true(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		for(auto &i : candidate) {
			double sum = 0;
			for(size_t t = 0; t < w.size(); t++)
				sum += w[t] * M.at(t,i);
			g_val[i] = sum;
		}
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
}; // }}}

// Greedy MIPS with Max-Heap Queue: An Implementation of Algorithm 3 with Max Heap
struct improved_heap_greedy_mips_t { // {{{
	typedef std::vector<size_t> candidate_t;
	typedef std::vector<size_t> pool_t; // sorted index array s[r]
	typedef std::vector<pool_t> pools_t;
	pools_t pools;

	// An Implementation of Algorithm 1 (CondIter)
	struct conditional_iterator { // {{{
		const pool_t *s;
		bool increasing;
		size_t idx;
		conditional_iterator() : s(NULL), increasing(true), idx(0) {}
		conditional_iterator(const pool_t& pool, val_type wt)
			: s(&pool), increasing(wt<0), idx(0) {}
		size_t current() const { return increasing? (*s)[idx] : (*s)[s->size()-idx-1];}
		size_t empty() const {return idx == s->size();}
		size_t get_next() {idx++; return current();}
	}; // }}}

	const dmat_t M;
	std::vector<size_t> g_cnt;
	std::vector<double> g_val;

	improved_heap_greedy_mips_t(const dmat_t &M): M(M.get_view()) { // {{{
		pools.resize(M.rows, pool_t(M.cols));
		g_cnt.resize(M.cols);
		g_val.resize(M.cols);

		for(size_t t = 0; t < M.rows; t++) {
			auto &Q = pools[t];
			for(size_t j = 0; j < M.cols; j++) {
				g_val[j] = M.at(t,j);
				Q[j] = j;
			}
			std::sort(Q.begin(), Q.end(), increasing_comp_t(g_val.data())); // increasing order
		}
		for(auto &v : g_val) v=0;
	} // }}}

	double search(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		candidate.clear();
		if(w.size() != pools.size()) {
			fprintf(stderr,"query with wrong size\n");
			return 0;
		}
		auto tic = std::clock();

		// Algorithm 2
		std::vector<conditional_iterator> iters(w.size());
		std::priority_queue<std::pair<val_type, size_t> > Q;
		for(size_t t = 0; t < w.size(); t++) {
			iters[t] = conditional_iterator(pools[t], w[t]);
			Q.push(std::make_pair(w[t]*M.at(t,iters[t].current()), t));
		}
		budget = std::min(budget, g_cnt.size());

		while(candidate.size() < budget) {
			auto top = Q.top();
			auto t = top.second;
			auto j = iters[t].current();
			Q.pop();
			if(g_cnt[j] == 0) {
				candidate.push_back(j);
				g_cnt[j] += 1;
			}
			while(!iters[t].empty()) {
				j = iters[t].get_next();
				if(g_cnt[j] == 0) {
					Q.push(std::make_pair(w[t]*M.at(t,j), t));
					break;
				}
			}
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
	double search_cnt(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		std::sort(candidate.begin(), candidate.end(), comparator<size_t>(g_cnt.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
	double search_val(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);

	} // }}}
	double search_true(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		for(auto &i : candidate) {
			double sum = 0;
			for(size_t t = 0; t < w.size(); t++)
				sum += w[t] * M.at(t,i);
			g_val[i] = sum;
		}
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}

}; // }}}

// positive entry only
struct sample_mips_t { // {{{
	typedef std::vector<size_t> candidate_t;
	const dmat_t M;
	typedef htree_t<val_type> pool_t;
	std::vector<pool_t> pools;
	std::vector<size_t> g_cnt;
	std::vector<double> g_val;
	rng_t rng;

	sample_mips_t(const dmat_t &M_) : M(M_.get_view()) { // {{{
		pools = std::vector<pool_t>(M.rows, pool_t(M.cols));
		g_cnt.resize(M.cols);
		g_val.resize(M.cols);
		for(size_t i = 0; i < M.rows; i++) {
			auto &Q = pools[i];
			for(size_t j = 0; j < M.cols; j++)
				Q[j] = M.at(i,j);
			Q.init_dense();
		}
	} // }}}
	double search(const dvec_t& w, size_t budget, candidate_t& candidate) { // {{{
		candidate.clear();
		if(w.size() != pools.size()) {
			fprintf(stderr,"query with wrong size\n");
			return 0;
		}
		auto tic = std::clock();
		htree_t<val_type> Ft(w.size());
		for(size_t t = 0; t < w.size(); t++) {
			auto &Q = pools[t];
			Ft[t] = w[t]*Q.total_sum();
		}
		Ft.init_dense();
		for(size_t b = 0; b < budget; b++) {
			auto urnd = rng.uniform()*Ft.total_sum();
			int t = Ft.log_sample(urnd);
			auto &Q = pools[t];
			urnd = rng.uniform()*Q.total_sum();
			auto i = Q.log_sample(urnd);
			if(i >= M.cols) printf("sample error");
			if(g_cnt[i] == 0)
				candidate.push_back(i);
			g_cnt[i] += 1;
		}
		return (std::clock()-tic)/(double)(CLOCKS_PER_SEC);

	} // }}}
	double search_cnt(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		std::sort(candidate.begin(), candidate.end(), comparator<size_t>(g_cnt.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock()-tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
	double search_true(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		for(auto &i : candidate) {
			double sum = 0;
			for(size_t t = 0; t < w.size(); t++)
				sum += w[t] * M.at(t,i);
			g_val[i] = sum;
		}
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
}; // }}}

// diamond sampling
struct diamond_mips_t { // {{{
	typedef std::vector<size_t> candidate_t;
	const dmat_t M;
	typedef htree_t<val_type> pool_t;
	std::vector<pool_t> pools;
	std::vector<size_t> g_cnt;
	std::vector<double> g_val;
	rng_t rng;

	diamond_mips_t(const dmat_t &M_) : M(M_.get_view()) { // {{{
		pools = std::vector<pool_t>(M.rows, pool_t(M.cols));
		g_cnt.resize(M.cols);
		g_val.resize(M.cols);
		for(size_t i = 0; i < M.rows; i++) {
			auto &Q = pools[i];
			for(size_t j = 0; j < M.cols; j++)
				Q[j] = fabs(M.at(i,j));
			Q.init_dense();
		}
	} // }}}
	double search(const dvec_t& w, size_t budget, candidate_t& candidate) { // {{{
		candidate.clear();
		if(w.size() != pools.size()) {
			fprintf(stderr,"query with wrong size\n");
			return 0;
		}
		auto tic = std::clock();
		htree_t<val_type> Ft(w.size()), Forig(w.size());
		for(size_t t = 0; t < w.size(); t++) {
			auto &Q = pools[t];
			Ft[t] = fabs(w[t])*Q.total_sum();
			Forig[t] = fabs(w[t]);
		}
		Ft.init_dense();
		for(size_t b = 0; b < budget; b++) {
			auto urnd = rng.uniform()*Ft.total_sum();
			int t = Ft.log_sample(urnd);
			auto &Q = pools[t];
			urnd = rng.uniform()*Q.total_sum();
			auto i = Q.log_sample(urnd);
			if(i >= M.cols) printf("sample error");
			urnd = rng.uniform()*Forig.total_sum();
			auto t2 = Forig.log_sample(urnd);
			if(g_cnt[i] == 0)
				candidate.push_back(i);
			g_val[i] += ((w[t]*M.at(i, t)*w[t2]) > 0? 1 : -1) * M.at(i, t2);
			g_cnt[i] += 1;
		}
		return (std::clock()-tic)/(double)(CLOCKS_PER_SEC);

	} // }}}
	double search_val(const dvec_t &w, size_t budget, candidate_t& candidate, size_t sample_ratio = 10) { // {{{
		auto tic = std::clock();
		search(w, sample_ratio*budget, candidate);
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		candidate.resize(budget);
		for(auto &i : candidate) {
			double sum = 0;
			for(size_t t = 0; t < w.size(); t++)
				sum += w[t] * M.at(t,i);
			g_val[i] = sum;
		}
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock()-tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
	double search_true(const dvec_t &w, size_t budget, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, budget, candidate);
		for(auto &i : candidate) {
			double sum = 0;
			for(size_t t = 0; t < w.size(); t++)
				sum += w[t] * M.at(t,i);
			g_val[i] = sum;
		}
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_cnt[i] = 0;
			g_val[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
}; // }}}

// return max_norm
template<typename T=double>
double mips_to_nn(const dmat_t &H, dmat_t &nnH) { // {{{
	nnH.lazy_resize(H.rows+1, H.cols, COLMAJOR);
	double max_norm2 = 0;
	for(size_t j = 0; j < H.cols; j++) {
		double tmp = 0;
		for(size_t t = 0; t < H.rows; t++) {
			nnH.at(t,j) = H.at(t,j);
			tmp += H.at(t,j)*H.at(t,j);
		}
		max_norm2 = std::max(tmp, max_norm2);
	}
	for(size_t j = 0; j < H.cols; j++) {
		double tmp = 0;
		for(size_t t = 0; t < H.rows; t++)
			tmp += H.at(t,j)*H.at(t,j);
		nnH.at(H.rows,j) = sqrt(max_norm2 - tmp);
	}
	return max_norm2;
} // }}}

// return rotation mean vector
template<typename T=double>
void normalize_transform(dmat_t &H, dmat_t &R, dvec_t &mean) { // {{{
	mean.resize(H.rows, 0);
	for(size_t i = 0; i < H.cols; i++)
		for(size_t t = 0; t < H.rows; t++)
			mean[t] += H.at(t,i);
	for(size_t t = 0; t < H.rows; t++)
		mean[t] /= H.cols;
	for(size_t i = 0; i < H.cols; i++)
		for(size_t t = 0; t < H.rows; t++)
			H.at(t,i)-= mean[t];
	dvec_t s;
	dmat_t V;
	svd(H, R, s, V);
	R.to_transpose();
	for(size_t i = 0; i < H.cols; i++)
		for(size_t t = 0; t < H.rows; t++)
			H.at(t,i) = V.at(i,t)*s[t];
} // }}}

struct pca_tree_t { // {{{
	size_t nr_candidates;
	size_t depth;
	std::vector<size_t> index;
	std::vector<size_t> start;
	std::vector<size_t> end;
	std::vector<double> pivot;
	pca_tree_t() {}
	pca_tree_t(const dmat_t& H_, size_t depth_) {init(H_, depth_);}
	void init(const dmat_t& H_, size_t depth_) { // {{{
		dmat_t H = H_.get_view(); H.to_rowmajor();
		depth = std::min(H.rows, depth_);
		index.resize(H.cols);
		for(size_t i = 0; i < H.cols; i++)
			index[i] = i;
		size_t tree_size = (1<<(depth+1)); // 2^(depth+1)
		start.resize(tree_size);
		end.resize(tree_size);
		pivot.resize(tree_size);
		start[0] = 0;
		end[0] = 2*index.size();
		for(size_t d = 0; d <= depth; d++) {
			for(size_t root = 1<<d; root < (1<<(d+1)); root++) {
				size_t parent = root >> 1;
				size_t mid = (start[parent]+end[parent])>>1;
				if(root==1 || (root&1)==0) {
					start[root] = start[parent];
					end[root] = mid;
				} else {
					start[root] = mid;
					end[root] = end[parent];
				}
				if(d < depth) {
					//auto Hd = H[H.rows-depth+d].data();
					auto Hd = H[d].data();
					//auto Hd = H[H.rows-d-1].data();
					//auto comp = increasing_comp_t(Hd);
					std::sort(index.begin()+start[root], index.begin()+end[root], increasing_comp_t(Hd));
					//std::sort(index.begin()+start[root], index.begin()+end[root]);
					//std::sort(&index[start[root]], &index[end[root]], comp);
					mid = ((start[root]+end[root]) >> 1);
					pivot[root] = Hd[index[mid]];
				}
			}
		}
	} // }}}
	size_t search_leaf(const dvec_t& w) const { // {{{
		size_t idx = 0;
		size_t root = 1;
		for(size_t d = 0; d < depth; d++) {
			idx <<= 1;
			if(w[d] >= pivot[root])
				idx |= 1;
		}
		return idx;
	} // }}}
	size_t neighbor_leaf(size_t idx, size_t d) const { // {{{
		return (idx ^ (1<<d));
	} // }}}
	dense_vector<size_t> get_conent(size_t idx) { // {{{
		idx += (1<<depth);
		return dense_vector<size_t>(end[idx]-start[idx], &index[start[idx]]);
	} // }}}
}; // }}}

struct pca_mips_t { // {{{
	typedef std::vector<size_t> candidate_t;
	const dmat_t M;
	std::vector<size_t> g_cnt;
	std::vector<double> g_val;
	dmat_t R;
	dvec_t mean;
	pca_tree_t pca_tree;
	bool normalized;

	pca_mips_t(const dmat_t&M_, size_t depth, bool normalized): M(M_.get_view()), normalized(normalized) { // {{{
		dmat_t MM;
		mips_to_nn(M_, MM);
		if(normalized)
			normalize_transform(MM, R, mean);
		if(depth >= MM.rows) depth = MM.rows;
		pca_tree.init(MM, depth);
		g_val.resize(MM.cols);
		g_cnt.resize(MM.cols);
	} // }}}
	double search(const dvec_t& w, candidate_t& candidate) { // {{{
		candidate.clear();
		if(w.size() != (M.rows)) {
			fprintf(stderr,"query with wrong size\n");
			return 0;
		}
		dvec_t w1(M.rows+1, 0.0), w2(M.rows+1, 0.0);
		if(normalized) {
			// w1 = w - mean
			for(size_t t = 0; t < w.size(); t++)
				w1[t] = w[t]-mean[t];
			w1[w.size()] = -mean[w.size()];

			// w2 = R*w1
			for(size_t t = 0; t < w2.size(); t++) {
				w2[t] = 0;
				for(size_t s = 0; s < w2.size(); s++)
					w2[t] += R.at(s,t)*w1[s];
			}
		} else {
			for(size_t t = 0; t < w.size(); t++)
				w2[t] = w[t];
		}

		auto tic = std::clock();
		size_t leaf = pca_tree.search_leaf(w2);
		auto tmp = pca_tree.get_conent(leaf);
		for(size_t idx = 0; idx < tmp.size(); idx++)
			candidate.push_back(tmp[idx]);
		for(size_t d = 0; d < pca_tree.depth; d++)  {
			auto neighbor = pca_tree.neighbor_leaf(leaf, d);
			auto tmp = pca_tree.get_conent(neighbor);
			for(size_t idx = 0; idx < tmp.size(); idx++) {
				if(g_cnt[tmp[idx]] > 0) continue;
				candidate.push_back(tmp[idx]);
				g_cnt[tmp[idx]] ++;
			}
		}
		return (std::clock()-tic)/(double)(CLOCKS_PER_SEC);

	} // }}}
	double search_true(const dvec_t &w, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, candidate);
		for(auto &i : candidate) {
			double sum = 0;
			for(size_t t = 0; t < w.size(); t++)
				sum += w[t] * M.at(t,i);
			g_val[i] = sum;
		}
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_val[i] = 0;
			g_cnt[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
}; // }}}

struct lsh_mips_t { // {{{
	typedef std::vector<size_t> candidate_t;
	const dmat_t M;
	size_t B, R; // amplification parameter for LSH
	dmat_t random_matrix;
	std::vector<std::vector<candidate_t>> hash_bins;
	std::vector<size_t> g_cnt;
	std::vector<double> g_val;
	lsh_mips_t(const dmat_t &M_, size_t B=1, size_t R=1): M(M_.get_view()), B(B), R(R) { // {{{
		dmat_t MM;
		mips_to_nn(M_, MM);
		g_val.resize(MM.cols);
		g_cnt.resize(MM.cols);
		hash_bins.resize(B, std::vector<candidate_t>(1<<R));
		random_matrix = drandn<val_type>(B*R, MM.rows, ROWMAJOR);
		dmat_t tmp = random_matrix * MM;
		for(size_t i = 0; i < MM.cols; i++) {
			for(size_t b = 0; b < B; b++) {
				size_t key = 0;
				for(size_t r = 0; r < R; r++) {
					key <<= 1;
					if(tmp.at(b*R+r, i) > 0) key |= 1;
				}
				hash_bins[b][key].push_back(i);
			}
		}
	} // }}}
	double search(const dvec_t& w, candidate_t& candidate) { // {{{
		candidate.clear();
		if(w.size() != (M.rows)) {
			fprintf(stderr,"query with wrong size\n");
			return 0;
		}
		dvec_t w2(M.rows+1, 0.0);
		for(size_t t = 0; t < w.size(); t++)
			w2[t] = w[t];
		auto tic = std::clock();
		for(size_t b = 0; b < B; b++) {
			size_t key = 0;
			for(size_t r = 0; r < R; r++) {
				key <<= 1;
				double tmp = 0;
				for(size_t t = 0; t < w2.size(); t++)
					tmp += w2[t]*random_matrix.at(t, b*R+r);
				if(tmp > 0) key |= 1;
			}
			for(auto& idx : hash_bins[b][key]) {
				if(g_cnt[idx] == 0)
					candidate.push_back(idx);
				g_cnt[idx] += 1;
			}
		}
		return (std::clock()-tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
	double search_true(const dvec_t &w, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		search(w, candidate);
		for(auto &i : candidate) {
			double sum = 0;
			for(size_t t = 0; t < w.size(); t++)
				sum += w[t] * M.at(t,i);
			g_val[i] = sum;
		}
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		for(auto &i : candidate) {
			g_val[i] = 0;
			g_cnt[i] = 0;
		}
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}

}; // }}}

struct naive_mips_t { // {{{
	typedef std::vector<size_t> candidate_t;
	const dmat_t M;
	std::vector<double> g_val;
	naive_mips_t(const dmat_t &M_): M(M_.get_view()), g_val(M_.cols){}
	double search_true(const dvec_t &w, candidate_t& candidate) { // {{{
		auto tic = std::clock();
		candidate.resize(M.cols);
		g_val.resize(M.cols);
		for(size_t i = 0; i < M.cols; i++)
			candidate[i] = i;
		for(auto &i : candidate) {
			double sum = 0;
			for(size_t t = 0;t < w.size(); t++)
				sum += w[t]*M.at(t,i);
			g_val[i] = sum;
		}
		std::sort(candidate.begin(), candidate.end(), comparator<double>(g_val.data()));
		return (std::clock() - tic)/(double)(CLOCKS_PER_SEC);
	} // }}}
}; // }}}

#endif // FAST_PRED_H
