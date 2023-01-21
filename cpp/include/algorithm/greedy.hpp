#ifndef GREEDY_HPP
#define GREEDY_HPP

#include <cassert>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "../cached_gram_matrix.hpp"
#include "../timer.hpp"
#include "oracle.hpp"
#include "param.hpp"
#include "result.hpp"
#include "utility.hpp"

class GreedyResult
{
private:
    std::vector<OptionalElementValuePair> element_value_pairs;
    std::vector<Measurement> measurements;

public:
    bool finished;

    explicit GreedyResult(const int reserve_size = 0) : finished(false)
    {
        element_value_pairs.reserve(reserve_size + 1);
        measurements.reserve(reserve_size + 1);
        add(std::nullopt, 0.0, 0.0, 0, 0, 0);
    }

    void add(const std::optional<int> element,
             const double value,
             const double time,
             const int num_computed_entries_L,
             const int num_oracle_calls,
             const int num_computed_offdiagonals_V,
             const std::vector<std::pair<int, int>>& computed_offdiagonals_V = {})
    {
        element_value_pairs.push_back({element, value});
        measurements.push_back({time,
                                num_computed_entries_L,
                                num_oracle_calls,
                                num_computed_offdiagonals_V,
                                computed_offdiagonals_V});
    }

    int size() const
    {
        return measurements.size();
    }

    Result last() const
    {
        return (*this)[size() - 1];
    }

    Result operator[](const int k) const
    {
        assert(0 <= k);

        if(k >= size()) {
            return Result::unfinished();
        }

        std::vector<int> sol(k);
        for(int i = 0; i < k; ++i) {
            assert(element_value_pairs[i + 1].element);
            sol[i] = *element_value_pairs[i + 1].element;
        }

        std::vector<std::pair<int, int>> offdiagonals;
        for(const auto& measurement : measurements) {
            offdiagonals.insert(offdiagonals.end(),
                                measurement.computed_offdiagonals_V.begin(),
                                measurement.computed_offdiagonals_V.end());
        }

        return Result(true,
                      sol,
                      element_value_pairs[k].value,
                      measurements[k].time,
                      measurements[k].num_computed_entries_L,
                      measurements[k].num_oracle_calls,
                      measurements[k].num_computed_offdiagonals_V,
                      offdiagonals);
    }
};

template<class S, class O, class M>
GreedyResult greedy(M L, const int k, const Param& param = {})
{
    const int n = L.rows();
    assert(0 <= k && k <= n);

    GreedyResult result(k);
    const Timer timer;

    std::vector<int> T(n);
    std::iota(T.begin(), T.end(), 0);
    auto oracle   = O::construct(L, k, param.log_computed_offdiagonals_V);
    auto strategy = S::construct(oracle, T.begin(), T.end(), false);

    for(int t = 0; t < k; ++t) {
        const auto e = *strategy.pop_largest();
        oracle.add(e);

        const double time = timer.get();
        result.add(e,
                   oracle.get_value(),
                   time,
                   get_num_computed_entries(L),
                   oracle.get_num_oracle_calls(),
                   oracle.get_num_computed_offdiagonals_V(),
                   oracle.get_computed_offdiagonals_V());

        if(time > param.time_limit
           || oracle.get_value() == -std::numeric_limits<double>::infinity()) {
            return result;
        }

        if(param.log_computed_offdiagonals_V) {
            oracle.clear_computed_offdiagonals_V();
        }
    }

    result.finished = true;
    return result;
}

template<class S, class O, class M>
GreedyResult greedy(const M& L, const Param& param = {})
{
    return greedy<S, O>(L, L.cols(), param);
}

#endif
