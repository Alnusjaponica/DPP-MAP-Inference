#ifndef RANDOM_GREEDY_HPP
#define RANDOM_GREEDY_HPP

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>

#include <boost/random/uniform_int_distribution.hpp>

#include "../cached_gram_matrix.hpp"
#include "../timer.hpp"
#include "param.hpp"
#include "result.hpp"

template<class S, class O, class M>
Result random_greedy(M L, const int k, const Param& param = {})
{
    const int n = L.rows();
    assert(0 <= k && k <= n);

    const Timer timer;

    std::vector<int> T(n);
    std::iota(T.begin(), T.end(), 0);
    auto oracle   = O::construct(L, k, param.log_computed_offdiagonals_V);
    auto strategy = S::construct(oracle, T.begin(), T.end(), true);

    std::mt19937 engine(param.seed);
    boost::random::uniform_int_distribution rand(0, std::max(0, k - 1));  // Flawfinder: ignore

    for(int t = 0; t < k; ++t) {
        if(const auto e = strategy.pop_kth_largest(rand(engine))) {
            oracle.add(*e);
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
