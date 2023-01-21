#ifndef RESULT_HPP
#define RESULT_HPP

#include <algorithm>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

struct Measurement
{
    double time;
    int num_computed_entries_L;
    int num_oracle_calls;
    int num_computed_offdiagonals_V;
    std::vector<std::pair<int, int>> computed_offdiagonals_V;
};

struct Result : public Measurement
{
    bool finished;
    std::vector<int> solution;
    double value;

    explicit Result(const bool finished,
                    const std::vector<int>& solution,
                    const double value,
                    const double time,
                    const int num_computed_entries_L,
                    const int num_oracle_calls,
                    const int num_computed_offdiagonals_V,
                    const std::vector<std::pair<int, int>>& computed_offdiagonals_V = {})
      : Measurement{time,
                    num_computed_entries_L,
                    num_oracle_calls,
                    num_computed_offdiagonals_V,
                    computed_offdiagonals_V},
        finished(finished),
        solution(solution),
        value(value)
    {}

    const Result& last() const
    {
        return *this;
    }

    static Result unfinished()
    {
        return Result(false, {}, 0.0, 0.0, 0, 0, 0);
    }
};

#endif
