#include <algorithm>
#include <ctime>

#include "fast-pred.h"

//#include "sparse_matrix.h"


void do_rank_diff(const test_case_t& tc, int argc, char *argv[]) { // {{{
	typedef std::vector<size_t> candidate_t;
	auto &true_rank = tc.true_rank;
	auto &W = tc.W;
	auto &H = tc.H;
	size_t n = H.cols;
	size_t k = H.rows;
	size_t nr_threads = omp_get_num_procs();
	//nr_threads = 1;
	omp_set_num_threads(nr_threads);

	printf("nr-threads %ld\n", nr_threads);


	std::vector<dvec_t> pool_max_vals(nr_threads, dvec_t(n));
	std::vector<candidate_t> pool_candidates(nr_threads, candidate_t(n));
	std::vector<candidate_t> pool_greedy_ranking(nr_threads, candidate_t(n));
	std::vector<candidate_t> pool_avg_cnt(nr_threads, candidate_t(tc.topk));
	std::vector<candidate_t> pool_ans(nr_threads, candidate_t(tc.topk));

#pragma omp parallel for
	for(size_t i = 0; i < tc.W.rows; i++) {
		auto tid = omp_get_thread_num();
		auto &candidates = pool_candidates[tid];
		auto &greedy_ranking = pool_greedy_ranking[tid];
		auto &avg_cnt = pool_avg_cnt[tid];
		auto &max_vals = pool_max_vals[tid];
		auto &ans = pool_ans[tid];

		auto &w = tc.W[i];
		for(size_t j = 0; j < n; j++) {
			auto &hj = tc.H[j];
			candidates[j] = j;
			val_type val = w[0]*hj[0];
			for(size_t t = 1; t < k; t++)
				//val += w[t]*hj[t];
				val = std::max(val, w[t]*hj[t]);
			max_vals[j] = val;
		}
		std::sort(candidates.begin(), candidates.end(),
				comparator<val_type>(max_vals.data()));
		// greedy_ranking[candidates] = 0:(n-1);
		for(size_t j = 0; j < n; j++)
			greedy_ranking[candidates[j]] = j;

		auto *val = &true_rank.val_t[true_rank.row_ptr[i]];
		auto *idx = &true_rank.col_idx[true_rank.row_ptr[i]];
		for(size_t t = 0; t < tc.topk; t++)
			ans[tc.topk - val[t]] = idx[t];
		size_t tmp_max = 0;
		for(size_t t = 0;t < tc.topk; t++) {
			//tmp_max = std::max(tmp_max, greedy_ranking[ans[t]]); avg_cnt[t] += tmp_max;
			tmp_max += (greedy_ranking[ans[t]]+1); avg_cnt[t] += tmp_max;
			avg_cnt[t] += (size_t) fabs(greedy_ranking[ans[t]] - t);

			//printf("(%ld %ld) = (%g %g)\n", (size_t)ans[t], candidates[t], max_vals[ans[t]], max_vals[candidates[t]]);
		}
	}
	auto &avg_cnt = pool_avg_cnt[0];
	printf("n %ld k %ld\n", n, k);
	for(size_t t = 0; t < tc.topk; t++) {
		for(size_t tid = 1; tid < nr_threads; tid++)
			avg_cnt[t] += pool_avg_cnt[tid][t];
		//printf(" %g", avg_cnt[t]/(double)tc.W.rows);
		printf(" %g", (avg_cnt[t]/(double)tc.W.rows)/((t+1)*(t+2)*0.5));
		//printf(" %g", avg_cnt[t]/(double)tc.W.rows);
	}
	puts("");
} // }}}

void usage() { // {{{
	puts("./go data.tc solver option1 [option2 ...]");
	puts(" solver = naive");
	puts("        = lsh R1 B1 [R2 B2] ... ");
	puts("        = pca depth1 depth2 ....");
	puts("        = sample budget1 budget2 ...");
	puts("        = diamond budget1 budget2 ...");
	puts("        = Diamond-sample-ratio ratio1 budget1 [ratio2 budget2] ...");
	puts("        = greedy budget1 budget2 ...");
	puts("        = Greedy-improve budget1 budget2 ...");
	puts("        = Heap-improve-greedy budget1 budget2 ...");
	puts("        = rank-diff -> test only");
} // }}}

void do_naive(const test_case_t& tc, int argc, char *argv[]) { // {{{
	pmf_ranker_t ranker;
	info_t info;
	auto topk = tc.topk;
	naive_mips_t mips(tc.H);
	ranker.init_work_space(tc.H.cols, topk, info);
	double time = 0;
	for(size_t i = 0; i < tc.W.rows; i++) {
		time += mips.search_true(tc.W[i], info.sorted_idx);
		info.sorted_idx.resize(topk);
		ranker.eval_ith_row(tc.true_rank, i, topk, info);
	}
	printf("solver naive time %.10g", time); ranker.summarize(info).print_full(); puts("");
	fflush(stdout);
} // }}}

void do_lsh(const test_case_t& tc, int argc, char *argv[]) { // {{{
	pmf_ranker_t ranker;
	info_t info;
	auto topk = tc.topk;
	int opt_idx = 0;
	while (opt_idx < argc) {
		size_t B = atoi(argv[opt_idx++]);
		size_t R = atoi(argv[opt_idx++]);
		lsh_mips_t mips(tc.H, B, R);
		ranker.init_work_space(tc.H.cols, topk, info);
		double time = 0;
		for(size_t i = 0; i < tc.W.rows; i++) {
			time += mips.search_true(tc.W[i], info.sorted_idx);
			info.sorted_idx.resize(topk);
			ranker.eval_ith_row(tc.true_rank, i, topk, info);
		}
		printf("solver lsh time %.10g B %ld R %ld", time, B, R); ranker.summarize(info).print_full(); puts("");
		fflush(stdout);
	}
} // }}}

void do_pca(const test_case_t& tc, int argc, char *argv[], bool normalized=true) { // {{{
	pmf_ranker_t ranker;
	info_t info;
	auto topk = tc.topk;
	int opt_idx = 0;
	while (opt_idx < argc) {
		size_t depth = atoi(argv[opt_idx++]);
		pca_mips_t mips(tc.H, depth, normalized);
		ranker.init_work_space(tc.H.cols, topk, info);
		double time = 0;
		for(size_t i = 0; i < tc.W.rows; i++) {
			time += mips.search_true(tc.W[i], info.sorted_idx);
			info.sorted_idx.resize(topk);
			ranker.eval_ith_row(tc.true_rank, i, topk, info);
		}
		printf("solver pca time %.10g depth %ld", time, depth); ranker.summarize(info).print_full(); puts("");
		fflush(stdout);
	}
} // }}}

void do_sample(const test_case_t& tc, int argc, char *argv[]) { // {{{
	pmf_ranker_t ranker;
	info_t info;
	auto topk = tc.topk;
	sample_mips_t mips(tc.H);
	int opt_idx = 0;
	while (opt_idx < argc) {
		size_t budget = atoi(argv[opt_idx++]);
		ranker.init_work_space(tc.H.cols, topk, info);
		double time = 0;
		for(size_t i = 0; i < tc.W.rows; i++) {
			time += mips.search_true(tc.W[i], budget, info.sorted_idx);
			info.sorted_idx.resize(topk);
			ranker.eval_ith_row(tc.true_rank, i, topk, info);
		}
		printf("solver sample time %.10g budget %ld", time, budget); ranker.summarize(info).print_full(); puts("");
		fflush(stdout);
	}
} // }}}

void do_diamond(const test_case_t& tc, int argc, char *argv[]) { // {{{
	pmf_ranker_t ranker;
	info_t info;
	auto topk = tc.topk;
	diamond_mips_t mips(tc.H);
	int opt_idx = 0;
	while (opt_idx < argc) {
		size_t budget = atoi(argv[opt_idx++]);
		ranker.init_work_space(tc.H.cols, topk, info);
		double time = 0;
		for(size_t i = 0; i < tc.W.rows; i++) {
			time += mips.search_true(tc.W[i], budget, info.sorted_idx);
			info.sorted_idx.resize(topk);
			ranker.eval_ith_row(tc.true_rank, i, topk, info);
		}
		printf("solver diamond time %.10g budget %ld", time, budget); ranker.summarize(info).print_full(); puts("");
		fflush(stdout);
	}
} // }}}

void do_diamond_sample_ratio(const test_case_t& tc, int argc, char *argv[]) { // {{{
	pmf_ranker_t ranker;
	info_t info;
	auto topk = tc.topk;
	diamond_mips_t mips(tc.H);
	int opt_idx = 0;
	while (opt_idx < argc) {
		size_t sample_ratio = atoi(argv[opt_idx++]);
		size_t budget = atoi(argv[opt_idx++]);
		ranker.init_work_space(tc.H.cols, topk, info);
		double time = 0;
		for(size_t i = 0; i < tc.W.rows; i++) {
			time += mips.search_val(tc.W[i], budget, info.sorted_idx, sample_ratio);
			info.sorted_idx.resize(topk);
			ranker.eval_ith_row(tc.true_rank, i, topk, info);
		}
		printf("solver diamond time %.10g budget %ld", time, budget); ranker.summarize(info).print_full(); puts("");
		fflush(stdout);
	}
} // }}}

void do_greedy(const test_case_t& tc, int argc, char *argv[]) { // {{{
	pmf_ranker_t ranker;
	info_t info;
	auto topk = tc.topk;
	greedy_mips_t mips(tc.H);
	int opt_idx = 0;
	while (opt_idx < argc) {
		size_t budget = atoi(argv[opt_idx++]);
		ranker.init_work_space(tc.H.cols, topk, info);
		double time = 0;
		for(size_t i = 0; i < tc.W.rows; i++) {
			time += mips.search_true(tc.W[i], budget, info.sorted_idx);
			info.sorted_idx.resize(topk);
			ranker.eval_ith_row(tc.true_rank, i, topk, info);
		}
		printf("solver greedy time %.10g budget %ld", time, budget); ranker.summarize(info).print_full(); puts("");
		fflush(stdout);
	}
} // }}}

void do_improved_greedy(const test_case_t& tc, int argc, char *argv[]) { // {{{
	pmf_ranker_t ranker;
	info_t info;
	auto topk = tc.topk;
	improved_greedy_mips_t mips(tc.H);
	int opt_idx = 0;
	while (opt_idx < argc) {
		size_t budget = atoi(argv[opt_idx++]);
		ranker.init_work_space(tc.H.cols, topk, info);
		double time = 0;
		for(size_t i = 0; i < tc.W.rows; i++) {
			time += mips.search_true(tc.W[i], budget, info.sorted_idx);
			info.sorted_idx.resize(topk);
			ranker.eval_ith_row(tc.true_rank, i, topk, info);
		}
		printf("solver greedy time %.10g budget %ld", time, budget); ranker.summarize(info).print_full(); puts("");
		fflush(stdout);
	}
} // }}}

void do_improved_heap_greedy(const test_case_t& tc, int argc, char *argv[]) { // {{{
	pmf_ranker_t ranker;
	info_t info;
	auto topk = tc.topk;
	improved_heap_greedy_mips_t mips(tc.H);
	int opt_idx = 0;
	while (opt_idx < argc) {
		size_t budget = atoi(argv[opt_idx++]);
		ranker.init_work_space(tc.H.cols, topk, info);
		double time = 0;
		for(size_t i = 0; i < tc.W.rows; i++) {
			time += mips.search_true(tc.W[i], budget, info.sorted_idx);
			info.sorted_idx.resize(topk);
			ranker.eval_ith_row(tc.true_rank, i, topk, info);
		}
		printf("solver greedy time %.10g budget %ld", time, budget); ranker.summarize(info).print_full(); puts("");
		fflush(stdout);
	}
} // }}}

// ./go data.tc solver
int main(int argc, char *argv[]) { // {{{
	bool normalized = true; // for pca
	if(argc == 1) { usage(); return 1; }
	test_case_t tc;
	tc.load_from_binary(argv[1]);
	switch (argv[2][0]) {
		case 'n':
			do_naive(tc, argc-3, &argv[3]);
			break;
		case 'l':
			do_lsh(tc, argc-3, &argv[3]);
			break;
		case 'p':
			do_pca(tc, argc-3, &argv[3], normalized);
			break;
		case 's':
			do_sample(tc, argc-3, &argv[3]);
			break;
		case 'd':
			do_diamond(tc, argc-3, &argv[3]);
			break;
		case 'D':
			do_diamond(tc, argc-3, &argv[3]);
			break;
		case 'g':
			do_greedy(tc, argc-3, &argv[3]);
			break;
		case 'G':
			do_improved_greedy(tc, argc-3, &argv[3]);
			break;
		case 'H':
			do_improved_heap_greedy(tc, argc-3, &argv[3]);
			break;
		case 'r':
			do_rank_diff(tc, argc-3, &argv[3]);
			break;
		default:
			printf("solver %s not supported!", argv[2]);
	};
} // }}}


