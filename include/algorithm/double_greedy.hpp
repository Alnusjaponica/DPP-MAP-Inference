#ifndef DOUBLE_GREEDY_HPP
#define DOUBLE_GREEDY_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <boost/random/bernoulli_distribution.hpp>

#include "../cached_gram_matrix.hpp"
#include "../timer.hpp"
#include "param.hpp"
#include "result.hpp"
#include "utility.hpp"

template<class O>
Result double_greedy(const Eigen::MatrixXd& L, const Eigen::MatrixXd& L_inv, const Param param = {})
{
    const int n = L.rows();
    const Timer timer;

    std::mt19937 engine(param.seed);

    auto oracle     = O::construct(L, n);
    auto oracle_inv = O::construct(L_inv, n);

    for(int i = 0; i < n; ++i) {
        const double v      = oracle.compute_marginal_gain_exponential(i);
        const double mg     = v > 1.0 ? std::log(v) : 0.0;
        const double v_inv  = oracle_inv.compute_marginal_gain_exponential(i);
        const double mg_inv = v_inv > 1.0 ? std::log(v_inv) : 0.0;
        const double p      = (mg == 0.0 && mg_inv == 0.0) ? 1.0 : mg / (mg + mg_inv);

        if(boost::random::bernoulli_distribution{p}(engine)) {  // FlawFinder: ignore
            oracle.add(i);
        }
        else {
            oracle_inv.add(i);
        }

        if(timer.get() > param.time_limit) {
            return Result(false, {}, 0.0, 0.0, 0, 0, 0);
        }
    }

    return Result(
        true,
        oracle.get_solution(),
        oracle.get_value(),
        timer.get(),
        0,
        oracle.get_num_oracle_calls() + oracle_inv.get_num_oracle_calls(),
        oracle.get_num_computed_offdiagonals_V() + oracle_inv.get_num_computed_offdiagonals_V());
}

template<class O>
Result double_greedy(const Eigen::MatrixXd& L, const Param param = {})
{
    if(const auto L_inv = inverse(L)) {
        return double_greedy<O>(L, *L_inv, param);
    }
    else {
        std::cerr << "L is singular." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#endif
