#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "../cached_gram_matrix.hpp"
#include "../strategy.hpp"
#include "../timer.hpp"
#include "param.hpp"
#include "result.hpp"

class InterlaceResult
{
private:
    std::vector<std::array<OptionalElementValuePair, 4>> quadruples;
    std::vector<Measurement> measurements;

    std::tuple<int, int, double> get_max(const int k) const
    {
        assert(0 <= k && k < int(size()));
        ElementValuePair maxima[4];

        for(int f = 0; f < 4; ++f) {
            maxima[f].value = quadruples[k][f].value;
            maxima[f].element =
                std::partition_point(quadruples.begin(),
                                     quadruples.begin() + k,
                                     [&](const auto& q) { return q[f].value < maxima[f].value; })
                - quadruples.begin();
        }

        const int f = std::max_element(maxima, maxima + 4) - maxima;
        return {f, maxima[f].element, maxima[f].value};
    }

public:
    bool finished = false;

    InterlaceResult(const int reserve_size = 0) : finished(false)
    {
        quadruples.reserve(reserve_size + 1);
        measurements.reserve(reserve_size + 1);
        add({std::nullopt, std::nullopt, std::nullopt, std::nullopt},
            {0.0, 0.0, 0.0, 0.0},
            0.0,
            0,
            0,
            0);
    }

    void add(const std::array<std::optional<int>, 4>& elements,
             const std::array<double, 4>& values,
             const double time,
             const int num_computed_entries_L,
             const int num_oracle_calls,
             const int num_computed_offdiagonals_V,
             const std::vector<std::pair<int, int>>& computed_offdiagonals_V = {})
    {
        quadruples.push_back({OptionalElementValuePair{elements[0], values[0]},
                              OptionalElementValuePair{elements[1], values[1]},
                              OptionalElementValuePair{elements[2], values[2]},
                              OptionalElementValuePair{elements[3], values[3]}});
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

        const auto [f, t, v] = get_max(k);
        std::vector<int> sol(t);
        for(int i = 0; i < t; ++i) {
            assert(quadruples[i + 1][f].element);
            sol[i] = *quadruples[i + 1][f].element;
        }

        return Result(true,
                      sol,
                      v,
                      measurements[k].time,
                      measurements[k].num_computed_entries_L,
                      measurements[k].num_oracle_calls,
                      measurements[k].num_computed_offdiagonals_V);
    }
};

template<class S, class O>
std::pair<std::optional<int>, std::optional<int>> interlace_subroutine(S& strategy_A,
                                                                       S& strategy_B,
                                                                       O& oracle_A,
                                                                       O& oracle_B)
{
    const auto e_a = strategy_A.pop_largest();
    if(e_a) {
        oracle_A.add(*e_a);
        strategy_B.remove(*e_a);
    }

    const auto e_b = strategy_B.pop_largest();
    if(e_b) {
        oracle_B.add(*e_b);
        strategy_A.remove(*e_b);
    }

    return {e_a, e_b};
}

template<class S, class O, class M>
InterlaceResult interlace_greedy(M L, const int k, const Param& param = {})
{
    const int n = L.rows();
    assert(0 <= k && k <= n);

    InterlaceResult result(k);
    const Timer timer;

    std::vector<int> T(n);
    std::iota(T.begin(), T.end(), 0);

    std::array oracles    = {O::construct(L, k, param.log_computed_offdiagonals_V),
                             O::construct(L, k, param.log_computed_offdiagonals_V),
                             O::construct(L, k, param.log_computed_offdiagonals_V),
                             O::construct(L, k, param.log_computed_offdiagonals_V)};
    std::array strategies = {S::construct(oracles[0], T.begin(), T.end(), true),
                             S::construct(oracles[1], T.begin(), T.end(), true),
                             S::construct(oracles[2], T.begin(), T.end(), true),
                             S::construct(oracles[3], T.begin(), T.end(), true)};

    for(int t = 0; t < k; ++t) {
        const auto e_01 =
            interlace_subroutine(strategies[0], strategies[1], oracles[0], oracles[1]);

        std::pair<std::optional<int>, std::optional<int>> e_23;
        if(t == 0) {
            e_23 = {strategies[2].pop_largest(), strategies[3].pop_largest()};
            if(e_23.first) {
                oracles[2].add(*e_23.first);
            }
            if(e_23.second) {
                oracles[3].add(*e_23.second);
            }
        }
        else {
            e_23 = interlace_subroutine(strategies[2], strategies[3], oracles[2], oracles[3]);
        }

        const double time = timer.get();
        result.add({e_01.first, e_01.second, e_23.first, e_23.second},
                   {
                       oracles[0].get_value(),
                       oracles[1].get_value(),
                       oracles[2].get_value(),
                       oracles[3].get_value(),
                   },
                   time,
                   get_num_computed_entries(L),
                   oracles[0].get_num_oracle_calls() + oracles[1].get_num_oracle_calls()
                       + oracles[2].get_num_oracle_calls() + oracles[3].get_num_oracle_calls(),
                   oracles[0].get_num_computed_offdiagonals_V()
                       + oracles[1].get_num_computed_offdiagonals_V()
                       + oracles[2].get_num_computed_offdiagonals_V()
                       + oracles[3].get_num_computed_offdiagonals_V());

        if(time > param.time_limit) {
            return result;
        }
    }

    result.finished = true;
    return result;
}

template<class S, class O, class M>
InterlaceResult interlace_greedy(const M& L, const Param& param = {})
{
    return interlace_greedy<S, O>(L, L.cols(), param);
}

#endif
