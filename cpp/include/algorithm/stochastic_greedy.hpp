#ifndef STOCHASTIC_GREEDY_HPP
#define STOCHASTIC_GREEDY_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <boost/random/uniform_int_distribution.hpp>

#include "../cached_gram_matrix.hpp"
#include "../timer.hpp"
#include "param.hpp"
#include "result.hpp"

inline void swap_T(const int i, const int j, std::vector<int>& T, std::vector<int>& T_inv)
{
    std::swap(T[i], T[j]);
    std::swap(T_inv[T[i]], T_inv[T[j]]);
}

inline void remove_T(const int e, std::vector<int>& T, std::vector<int>& T_inv)
{
    swap_T(T_inv[e], T.size() - 1, T, T_inv);
    T_inv[e] = -1;
    T.pop_back();
}

template<class R>
void fisher_yates_shuffle(std::vector<int>& T, std::vector<int>& T_inv, const int k, R& engine)
{
    const int n = T.size();
    for(int i = 0; i < k; i++) {
        const int j =
            boost::random::uniform_int_distribution{i, n - 1}(engine);  // FlawFinder: ignore
        swap_T(i, j, T, T_inv);
    }
}

template<class S, class O, class M>
Result stochastic_greedy(M L, const int k, const Param& param = {})
{
    const double eps = 0.5;

    const int n = L.rows();
    assert(0 <= k && k <= n);

    const Timer timer;

    std::mt19937 engine(param.seed);
    const int s = std::ceil(double(n) / double(k) * std::log(1.0 / eps));

    auto oracle = O::construct(L, k, param.log_computed_offdiagonals_V);

    std::vector<int> T(n);
    std::iota(T.begin(), T.end(), 0);
    auto T_inv = T;

    for(int t = 0; t < k; ++t) {
        const int current_s = std::min(s, n - int(oracle.get_solution().size()));
        fisher_yates_shuffle(T, T_inv, current_s, engine);
        auto strategy = S::construct(oracle, T.begin(), T.begin() + current_s, false);
        if(const auto e = strategy.pop_largest()) {
            oracle.add(*e);
            remove_T(*e, T, T_inv);
        }

        if(timer.get() > param.time_limit) {
            return Result(false, {}, 0.0, 0.0, 0, 0, 0);
        }
    }

    return Result(true,
                  oracle.get_solution(),
                  oracle.get_value(),
                  timer.get(),
                  get_num_computed_entries(L),
                  oracle.get_num_oracle_calls(),
                  oracle.get_num_computed_offdiagonals_V(),
                  oracle.get_computed_offdiagonals_V());
}

#endif
