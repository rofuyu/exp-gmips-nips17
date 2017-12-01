#ifndef _PMF_H_
#define _PMF_H_


// {{{ Headers
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>
#include <cmath>
#include <assert.h>
#include <vector>
#include <limits>
#include <omp.h>

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#define puts(str) mexPrintf("%s\n",(str))
#define fflush(str) mexEvalString("drawnow")
#endif

#include "sparse_matrix.h"
#include "block_matrix.h"

#ifdef _USE_FLOAT_
#define val_type float
#else
#define val_type double
#endif

typedef dense_vector<val_type> dvec_t;
typedef dense_matrix<val_type> dmat_t;
typedef identity_matrix<val_type> eye_t;
typedef sparse_matrix<val_type> smat_t;
typedef block_matrix<val_type> blocks_t;
typedef general_matrix<val_type> gmat_t;
//}}}

// pmf model
class pmf_model_t {//{{{
	public:
		size_t rows, cols;
		size_t k;
		dmat_t W, H;
		val_type global_bias;
		major_t major_type;

		pmf_model_t(major_t major_type_=default_major): major_type(major_type_){}
		pmf_model_t(size_t rows_, size_t cols_, size_t k_, major_t major_type_, bool do_rand_init=true, val_type global_bias_=0.0);

		pmf_model_t(const dmat_t &W, const dmat_t &H, val_type global_bias=0.0) : rows(W.rows), cols(H.rows), k(W.cols), W(W.get_view()), H(H.get_view()), global_bias(global_bias) { // {{{
			assert(W.cols == H.cols);
			if(W.is_rowmajor() && H.is_rowmajor())
				major_type = ROWMAJOR;
			else if(W.is_colmajor() && H.is_colmajor())
				major_type = COLMAJOR;
			else {
				major_type = ROWMAJOR;
				this->W.to_rowmajor();
				this->H.to_rowmajor();
			}
		} // }}}

		void rand_init(long seed=0L);

		val_type predict_entry(size_t i, size_t j) const;
		template<typename T, typename T2>
		void predict_entries(size_t nr_entries, const T *row_idx, const T *col_idx, T2 *pred_val, int idx_base=0, int nr_threads=0) const { // {{{
			if (nr_threads == 0)
				nr_threads = omp_get_max_threads();
			omp_set_num_threads(nr_threads);
#pragma omp parallel for schedule(static)
			for(long i = 0; i < nr_entries; i++) {
				pred_val[i] = (T2) predict_entry((size_t)row_idx[i]-idx_base, (size_t)col_idx[i]-idx_base);
			}
		} // }}}

		template<typename T, typename T2>
		void predict_row(size_t r, size_t nr_entries, T *col_idx, T2 *pred_val, int idx_base=0) const { // {{{
			for(size_t i = 0; i < nr_entries; i++)  {
				size_t c = (size_t)(col_idx[i]-idx_base);
				pred_val[c] = predict_entry(r, c);
			}
		} // }}}

		template<typename T, typename T2>
		void predict_col(size_t c, size_t nr_entries, T *row_idx, T2 *pred_val, int idx_base=0) const { // {{{
			for(size_t i = 0; i < nr_entries; i++) {
				size_t r = (size_t)(row_idx[i]-idx_base);
				pred_val[r] = predict_entry(r, c);
			}
		} // }}}

		void apply_permutation(const std::vector<unsigned> &row_perm, const std::vector<unsigned> &col_perm);
		void apply_permutation(const unsigned *row_perm=NULL, const unsigned *col_perm=NULL);
		void save(FILE *fp);
		void load(FILE *fp, major_t major_type_);
	private:
		void mat_rand_init(dmat_t &X, size_t m, size_t n, long seed);
};//}}}

// ================ Ranking Evaluation Utility Functions ==================
// decreasing comparator
template<typename T>
struct decreasing_comp_t { // {{{
	const T *pred_val;
	decreasing_comp_t(const T *_val): pred_val(_val) {}
	bool operator()(const size_t i, const size_t j) const {return pred_val[j] < pred_val[i];}
}; // }}}

// input: pred_val is a double array of length=len
//        idx is an size_t array of length=len with 0,1,...,len-1 as its elements
// output: the topk elements of idx is sorted according the decreasing order of elements of pred_val.
template<typename T>
void sort_idx_by_val(const T *pred_val, size_t len, size_t *idx, size_t topk) { // {{{
	size_t *mid = idx+(topk > len? len : topk);
	std::partial_sort(idx, mid, idx+len, decreasing_comp_t<T>(pred_val));
} // }}}

// Initialize pred_val to -inf for ignored indices and 0 for others
// Initialize candidates with 0,...,len-1
// return valid_len of candidates
template<typename T>
size_t pmf_prepare_candidates(size_t len, double *pred_val, size_t *candidates, size_t &valid_len, size_t nr_ignored = 0, T *ignored_list=NULL) {  // {{{
	const double Inf = std::numeric_limits<double>::infinity();
	for(size_t i = 0; i < len; i++) {
		pred_val[i] = 0;
		candidates[i] = i;
	}
	if(nr_ignored != 0 && ignored_list != NULL)
		for(size_t i = 0; i < nr_ignored; i++) {
			long ignored_idx = (long) ignored_list[i];
			if(ignored_idx >= 0 && ignored_idx < len)
				pred_val[(long)ignored_list[i]] = -Inf;
		}

	valid_len = len;
	for(size_t i = 0; i < valid_len; i++)
		if(pred_val[candidates[i]] < 0) {
			std::swap(candidates[i], candidates[valid_len-1]);
			valid_len--;
			i--;
		}
	return valid_len;
} // }}}
template<typename T>
size_t pmf_prepare_candidates(size_t len, double *pred_val, size_t *candidates, const T& ignored_list = T()) {  // {{{
	size_t valid_len = 0;
	return pmf_prepare_candidates(len, pred_val, candidates, valid_len, ignored_list.size(), ignored_list.data());
} // }}}

template<typename T>
inline double gain(T rel) { return static_cast<T>(exp2(rel)-1); }

template<typename T>
inline double discount(T l) { return 1.0/log2(l+2); }

// input: idx is an sorted index array of length=len
// output: dcg is the array of length=topk with accumuated dcg information
// return: dcg@topk
template<typename T>
T compute_dcg(const T *true_rel, size_t *sorted_idx, size_t len, int topk, T *dcg=NULL) { // {{{
	int levels = topk>len? len : topk;
	T cur_dcg = 0.0;
	for(int l = 0; l < levels; l++) {
		cur_dcg += gain(true_rel[sorted_idx[l]]) * discount(l);
		if(dcg)
			dcg[l] = cur_dcg;
	}
	if(dcg)
		for(int l = levels; l < topk; l++)
			dcg[l] = cur_dcg;
	return cur_dcg;
} // }}}

struct info_t { // {{{
	std::vector<size_t> sorted_idx;
	std::vector<double> true_rel;
	std::vector<double> pred_val;
	std::vector<double> tmpdcg, maxdcg;
	std::vector<double> dcg, ndcg;
	std::vector<double> prec, recall, tmp_prec;
	std::vector<size_t> count;
	double map, auc, hlu;
	void print(FILE *fp=stdout) const {
		fprintf(fp, " map %.5g auc %.5g hlu %.5g ", map, auc, hlu);
		for(size_t i = 0; i < std::min(ndcg.size(),size_t(5)); i+=1) fprintf(fp, " p@%ld %.5g", i+1, prec[i]);
		//for(size_t i = 0; i < std::min(ndcg.size(),size_t(5)); i+=2) fprintf(fp, " r@%ld %.5g", i+1, recall[i]);
		for(size_t i = 0; i < std::min(ndcg.size(),size_t(5)); i+=1) fprintf(fp, " n@%ld %.5g", i+1, ndcg[i]);
	}
	void print5(FILE *fp=stdout) const {
		fprintf(fp, " map %.5g auc %.5g hlu %.5g ", map, auc, hlu);
		size_t list[] = {0, 4, 9, 14, 19};
		for(size_t i = 0; i < 5; i+=1) if(list[i] < prec.size()) fprintf(fp, " p@%ld %.5g", list[i]+1, prec[i]);
		for(size_t i = 0; i < 5; i+=1) if(list[i] < ndcg.size()) fprintf(fp, " n@%ld %.5g", list[i]+1, ndcg[i]);
	}
	void print_full(FILE *fp=stdout) const {
		fprintf(fp, " map %.5g auc %.5g hlu %.5g ", map, auc, hlu);
		for(size_t i = 0; i < prec.size(); i+=1) fprintf(fp, " p@%ld %.5g", i+1, prec[i]);
		for(size_t i = 0; i < ndcg.size(); i+=1) fprintf(fp, " n@%ld %.5g", i+1, ndcg[i]);
	}
}; // }}}

struct rank_entry_t{ // {{{
	size_t i,j,rank,nr_pos;
	rank_entry_t(size_t i=0, size_t j=0, size_t rank=0, size_t nr_pos=0):
		i(i), j(j), rank(rank), nr_pos(nr_pos) {}
}; // }}}

class pmf_ranker_t { // {{{
	protected:
		typedef std::vector<rank_entry_t> rank_vec_t;
		typedef dense_vector<unsigned> ig_dvec_t; // ignored_list
		std::vector<info_t> info_set;
		std::vector<rank_vec_t> rank_vec_set;

		// init space for multiple threads
		void init_work_space(const smat_t& testR, int topk, int threads) { // {{{
			const size_t &cols = testR.cols;
			info_set.clear(); info_set.resize(threads);
			rank_vec_set.clear(); rank_vec_set.resize(threads);
			for(int th = 0; th < threads; th++)
				init_work_space(cols, topk, info_set[th]);
		} // }}}

		// aggregate results from multiple threads
		template<typename T>
		info_t aggregate_results(int idx_base, T *pos_rank=(unsigned*)NULL) { // {{{
			size_t nr_total_pos = rank_vec_set[0].size();
			info_t &final_info = info_set[0];
			size_t topk = final_info.prec.size();
			for(size_t th = 1; th < info_set.size(); th++) {
				nr_total_pos += rank_vec_set[th].size();
				info_t &info = info_set[th];
				for(size_t t = 0; t < topk; t++) {
					final_info.dcg[t] += info.dcg[t];
					final_info.ndcg[t] += info.ndcg[t];
					final_info.count[t] += info.count[t];
					final_info.prec[t] += info.prec[t];
					final_info.recall[t] += info.recall[t];
				}
				final_info.map += info.map;
				final_info.auc += info.auc;
				final_info.hlu += info.hlu;
				final_info.count[topk] += info.count[topk];
			}

			if(pos_rank!=NULL) {
				size_t idx = 0;
				for(size_t tid = 0; tid < rank_vec_set.size(); tid++) {
					rank_vec_t &rank_vec = rank_vec_set[tid];
					for(size_t s = 0; s < rank_vec.size(); s++) {
						// change everything from 0-based to 1-based
						pos_rank[idx+0*nr_total_pos] = rank_vec[s].i+idx_base;
						pos_rank[idx+1*nr_total_pos] = rank_vec[s].j+idx_base;
						pos_rank[idx+2*nr_total_pos] = rank_vec[s].rank+idx_base;
						pos_rank[idx+3*nr_total_pos] = rank_vec[s].nr_pos+idx_base;
						idx++;
					}
				}
			}
			return summarize(final_info);
		} // }}}

	public:
		double neutral_rel, halflife; // parameters for HLU

		pmf_ranker_t(double neutral_rel=0, double halflife=5): neutral_rel(neutral_rel), halflife(halflife) {}

		// Evaluation utility for single row // {{{
		//    info_t info;
		//    ranker.init_work_space(testR.cols, topk, info);
		//    ranker.eval_ith_row(testR, 0, topk, info);
		//              .....
		//    ranker.eval_ith_row(testR, testR.rows, topk, info);
		//    ranker.summarize(info)
		info_t& init_work_space(size_t cols, int topk, info_t &info) { // {{{
			info.sorted_idx.clear(); info.sorted_idx.reserve(cols);
			info.true_rel.clear(); info.true_rel.resize(cols, 0);
			info.pred_val.clear(); info.pred_val.resize(cols, 0);

			info.tmpdcg.clear(); info.tmpdcg.resize(topk);
			info.maxdcg.clear(); info.maxdcg.resize(topk);
			info.dcg.clear(); info.dcg.resize(topk);
			info.ndcg.clear(); info.ndcg.resize(topk);
			info.prec.clear(); info.prec.resize(topk);
			info.recall.clear(); info.recall.resize(topk);
			info.tmp_prec.clear(); info.tmp_prec.resize(topk);
			info.count.clear(); info.count.resize(topk+1);
			info.map = 0;
			info.auc = 0;
			info.hlu = 0;
		} // }}}

		// Evaluation on a single row with sparse true relevance  {{{
		//   nz_row/true_idx/true_rel: sparse vector for true relevance
		//   topk: topk evaluation
		//   info.sorted_idx: the candidates to be evaluated (length = size of sorted candidates)
		//   info.true_rel: dense space for the entire true relevance (length = size of total candidates)
		//   rank_vec: a container for rank_entry_t incurred for this evaluation if not null.
		// Output:
		//   info.sorted_idx: will be changed
		//   info.map/auc/hlu/ndcg/maxndcg/prec/recall/count will be updated. // }}}
		template<typename T1, typename T2>
		void eval_sparse_true_rel(size_t nz_row, const T1 *true_idx, const T2* true_rel, int topk, info_t &info, int i=0, rank_vec_t *rank_vec = NULL) { // {{{
			size_t cols = info.true_rel.size();
			if(nz_row == 0) return;
			for(size_t idx = 0; idx < nz_row; idx++)
				info.true_rel[true_idx[idx]] += (true_rel!=NULL? true_rel[idx]: 1.0);

			// MAP & AUC & HLU & PREC & RECALL  // {{{
			double localmap = 0;
			double localauc = 0;
			double localhlu = 0, localhlu_max = 0;
			size_t neg_cnt = 0, pos_cnt = 0, violating_pairs = 0;
			size_t valid_len = info.sorted_idx.size();
			for(int t = 0; t < topk; t++) info.tmp_prec[t] = 0;
			for(size_t j = 0; j < valid_len; j++) {
				size_t col = info.sorted_idx[j];
				if(info.true_rel[col] > 0) {
					// j is the rank of this item
					// pos_cnt is the number of "positive" items ranked before this item
					if(rank_vec != NULL)
						rank_vec->push_back(rank_entry_t(i, col, j, pos_cnt));
					localhlu += (info.true_rel[col]-neutral_rel)*pow(0.5,(j)/(halflife-1.0));
					localhlu_max += (info.true_rel[col]-neutral_rel)*pow(0.5,(pos_cnt)/(halflife-1.0));

					pos_cnt += 1;
					localmap += 100*(double)pos_cnt/(double)(j+1);
					violating_pairs += neg_cnt;
				} else {
					neg_cnt += 1;
				}
				if(j < topk) {
					info.tmp_prec[j] = pos_cnt;
					info.prec[j] += 100*pos_cnt;
				}
			}
			if(pos_cnt > 0) {
				for(int t = 0; t < topk; t++) {
					//info.prec[t] += 100*info.tmp_prec[t]/(t+1.0);
					info.recall[t] += 100*info.tmp_prec[t]/pos_cnt;
				}
			}
			if(pos_cnt > 0)
				localmap /= (double) pos_cnt;
			if(pos_cnt > 0 && neg_cnt > 0)
				localauc = (double)(pos_cnt*neg_cnt-violating_pairs)/(double)(pos_cnt*neg_cnt);
			else
				localauc = 1;
			if(pos_cnt > 0 && localhlu_max > 0)
				localhlu = 100*localhlu/localhlu_max;
			if(valid_len > 0) {
				info.map += localmap;
				info.auc += localauc;
				info.hlu += localhlu;
				info.count[topk] ++;
			}
			// }}}

			// NDCG // {{{
			compute_dcg(info.true_rel.data(), info.sorted_idx.data(), valid_len, topk, info.tmpdcg.data());
			valid_len = nz_row;
			if(valid_len) {
				info.sorted_idx.resize(valid_len);
				size_t *sorted_idx = info.sorted_idx.data();
				for(size_t idx = 0; idx < nz_row; idx++)
					sorted_idx[idx] = (size_t) true_idx[idx];
				sort_idx_by_val(info.true_rel.data(), valid_len, sorted_idx, topk);
				compute_dcg(info.true_rel.data(), info.sorted_idx.data(), valid_len, topk, info.maxdcg.data());

				for(int k = 0; k < topk; k++) {
					double tmpdcg = info.tmpdcg[k];
					double tmpmaxdcg = info.maxdcg[k];
					if(std::isfinite(tmpdcg) && std::isfinite(tmpmaxdcg) && tmpmaxdcg>0) {
						info.dcg[k] += tmpdcg;
						info.ndcg[k] += 100*tmpdcg/tmpmaxdcg;
						info.count[k] ++;
					}
				}
			}
			// }}}
			for(size_t idx = 0; idx < nz_row; idx++)
				info.true_rel[true_idx[idx]] -= (true_rel!=NULL? true_rel[idx]: 1.0);

		} // }}}

		template<typename T1>
		void eval_sparse_true_rel(size_t nz_row, const T1 *true_idx, int topk, info_t &info, int i=0, rank_vec_t *rank_vec = NULL) { // {{{
			eval_sparse_true_rel(nz_row, true_idx, (T1*)(NULL), topk, info, i, rank_vec);
		} // }}}

		// Output:
		//    info.sorted_idx: topk candidates
		//    info.pred_val: predicted model.cols values
		//    pred_topk[topk*i ~ topk*(i+1)]: topk candidates in type T1
#if defined(CPP11)
		template<typename T1=unsigned>
#else
		template<typename T1>
#endif
		void predict_ith_row(const pmf_model_t& model, int i, int topk, info_t &info, const ig_dvec_t& ignored_list=ig_dvec_t(), T1 *pred_topk=NULL, int idx_base=0) { // {{{
			info.sorted_idx.resize(model.cols);
			info.pred_val.resize(model.cols);
			size_t valid_len = pmf_prepare_candidates(model.cols, info.pred_val.data(), info.sorted_idx.data(), ignored_list);
			info.sorted_idx.resize(valid_len);
			model.predict_row(i, valid_len, info.sorted_idx.data(), info.pred_val.data());
			sort_idx_by_val(info.pred_val.data(), valid_len, info.sorted_idx.data(), valid_len);

			if(pred_topk!=NULL)
				for(int t = 0; t < topk; t++)
					pred_topk[topk*i + t] = (T1)(t<model.cols? (double) (info.sorted_idx[t] + idx_base): model.cols);
		} // }}}

		void preidct_single(const dvec_t& w, const dmat_t &H, int topk, info_t &info, const ig_dvec_t& ignored_list=ig_dvec_t()) { // {{{
			assert(w.size() == H.cols);
			predict_ith_row<unsigned>(pmf_model_t(dmat_t(w,ROWMAJOR),H), 0, topk, info, ignored_list);
		} // }}}

		void eval_ith_row(const smat_t& testR, size_t i, int topk, info_t &info, rank_vec_t *rank_vec = NULL) { // {{{
			size_t nz_row = testR.nnz_of_row(i);
			const unsigned *true_idx = &testR.col_idx[testR.row_ptr[i]];
			const val_type *true_rel = &testR.val_t[testR.row_ptr[i]];
			eval_sparse_true_rel(nz_row, true_idx, true_rel, topk, info, i, rank_vec);
		} // }}}

		// summarize the evaluation results accumulated in info
		info_t& summarize(info_t &info) { // {{{
			size_t topk = info.prec.size();
			if(topk == 0) return info;
			if(info.count[topk] > 0) {
				info.map /= (double) info.count[topk];
				info.auc /= (double) info.count[topk];
				info.hlu /= (double) info.count[topk];
				for(size_t t = 0; t < topk; t++) {
					info.prec[t] /= (double)(info.count[topk]*(t+1));
					info.recall[t] /= (double) info.count[topk];
				}
			}
			for(size_t t = 0; t < topk; t++) {
				info.dcg[t] = info.dcg[t] / (double) info.count[t];
				info.ndcg[t] = info.ndcg[t] / (double) info.count[t];
			}
			info.sorted_idx.clear();
			info.pred_val.clear();
			info.true_rel.clear();
			return info;
		} // }}}
		// }}}

		//   Evaluation utility for multiple rows  {{{
		// predict top candidates based on W*H'
		template<typename T1>
		void predict(const dmat_t &W, const dmat_t &H, int topk, T1 *pred_topk, const smat_t& ignored = smat_t(), major_t major_type=ROWMAJOR, int idx_base=0) { // {{{
			eval(smat_t(W.rows,H.rows), W, H, topk, ignored, pred_topk, (T1*)NULL, major_type, idx_base);
		} // }}}

		// evaluation on the top candidates predicted by W*H'
#if defined(CPP11)
		template<typename T1=unsigned, typename T2=unsigned>
#else
		template<typename T1, typename T2>
#endif
		info_t eval(const smat_t& testR, const dmat_t &W, const dmat_t &H, int topk, const smat_t& ignored=smat_t(), T1 *pred_topk=NULL, T2 *pos_rank=NULL, major_t major_type=ROWMAJOR, int idx_base=0) { // {{{
			assert(testR.rows == W.rows && testR.cols == H.rows && W.cols == H.cols);
			if(major_type == COLMAJOR)
				return eval(testR.transpose(), H, W, topk, ignored.transpose(), pred_topk, pos_rank, ROWMAJOR, idx_base);
			size_t rows = testR.rows, cols = testR.cols;
			pmf_model_t model(W, H);
			int threads = omp_get_max_threads();
			init_work_space(testR, topk, threads);
#pragma omp parallel for
			for(size_t i = 0; i < rows; i++) {
				if(testR.nnz_of_row(i) == 0 && pred_topk==NULL)
					continue;
				ig_dvec_t ignored_list;
				if(ignored.nnz>0 && ignored.nnz_of_row(i)>0)
					ignored_list = ig_dvec_t(ignored.nnz_of_row(i), &ignored.col_idx[ignored.row_ptr[i]]);

				int tid = omp_get_thread_num();
				info_t &info = info_set[tid];
				predict_ith_row(model, i, topk, info, ignored_list, pred_topk, idx_base);
				rank_vec_t *rank_vec = (pos_rank!=NULL)? &rank_vec_set[tid]: NULL;
				eval_ith_row(testR, i, topk, info, rank_vec);
			}
			return aggregate_results(idx_base, pos_rank);
		} // }}}

		// evaluation on the top candidates given by pred_topk
		template<typename T>
		info_t eval(const smat_t& testR, const T *pred_topk, int topk, major_t major_type = ROWMAJOR, int idx_base=0) { // {{{
			if(major_type == COLMAJOR)
				return eval(testR.transpose(), pred_topk, topk, ROWMAJOR, idx_base);
			// all the values in pred_topk - idx_base < rows
			size_t rows = testR.rows, cols = testR.cols;
			int threads = omp_get_max_threads();
			init_work_space(testR, topk, threads);
#pragma omp parallel for
			for(size_t i = 0; i < rows; i++) {
				int tid = omp_get_thread_num();
				info_t &info = info_set[tid];
				info.sorted_idx.clear();
				size_t valid_len = 0;
				for(int t = 0; t < topk; t++) {
					long tmp_idx = (long) (pred_topk[i*topk+t]-idx_base);
					if(0 <= tmp_idx && tmp_idx < cols) {
						info.sorted_idx.push_back(tmp_idx);
						valid_len ++;
					}
				}
				eval_ith_row(testR, i, topk, info);
			}
			// }}}

			return aggregate_results<unsigned>(idx_base);
		} // }}}
}; // }}}

#endif // end of _PMF_H
