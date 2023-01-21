#ifndef PARAM_HPP
#define PARAM_HPP

#include <limits>

struct Param
{
    double time_limit                = std::numeric_limits<double>::infinity();
    std::uint_fast32_t seed          = 0;
    bool log_computed_offdiagonals_V = false;
};

#endif
