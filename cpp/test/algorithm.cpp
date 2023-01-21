#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "algorithm/double_greedy.hpp"
#include "algorithm/greedy.hpp"
#include "algorithm/interlace_greedy.hpp"
#include "algorithm/random_greedy.hpp"
#include "algorithm/stochastic_greedy.hpp"
#include "cached_gram_matrix.hpp"
#include "io.hpp"
#include "oracle.hpp"
#include "strategy.hpp"
#include "utility.hpp"

using namespace std;
using namespace Eigen;
using namespace testing;

pair<vector<int>, double> sol_value(const Result& x)
{
    return {x.solution, x.value};
}

template<class M>
auto get_expected(const M& L, const vector<int>& solution)
{
    return Pair(solution, DoubleEq(logdet(L(solution, solution))));
}

TEST(greedy, empty)
{
    const MatrixXd B = MatrixXd::Zero(3, 0);
    const MatrixXd L = B.transpose() * B;
    const CachedGramMatrix C(B);
    const int k = 0;

    const auto expected = get_expected(L, {});

    EXPECT_THAT(sol_value(greedy<NonLazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<NonLazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<NonLazy, Oracle>(C, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<NonLazy, Fast>(C, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Oracle>(C, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Fast>(C, k).last()), expected);
}

TEST(greedy, small)
{
    const auto B     = gaussian(50, 20);
    const MatrixXd L = B.transpose() * B;
    const CachedGramMatrix C(B);
    const int k = 10;

    const auto expected = get_expected(L, {13, 18, 4, 3, 15, 7, 17, 19, 14, 2});

    EXPECT_THAT(sol_value(greedy<NonLazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<NonLazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<NonLazy, Oracle>(C, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<NonLazy, Fast>(C, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Oracle>(C, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Fast>(C, k).last()), expected);
}

TEST(greedy, negative)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;

    const int k         = 4;
    const auto expected = get_expected(L, {7, 2, 3, 0});

    EXPECT_THAT(sol_value(greedy<NonLazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<NonLazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Fast>(L, k).last()), expected);
}

TEST(greedy, all)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;

    const int k         = 8;
    const auto expected = get_expected(L, {7, 2, 3, 0, 4, 1, 5, 6});

    EXPECT_THAT(sol_value(greedy<NonLazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<NonLazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(greedy<Lazy, Fast>(L, k).last()), expected);
}

TEST(random_greedy, empty)
{
    const MatrixXd B = MatrixXd::Zero(3, 0);
    const MatrixXd L = B.transpose() * B;
    const CachedGramMatrix C(B);
    const int k = 0;

    const auto expected = get_expected(L, {});

    EXPECT_THAT(sol_value(random_greedy<NonLazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<NonLazy, Fast>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Fast>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<NonLazy, Oracle>(C, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<NonLazy, Fast>(C, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Oracle>(C, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Fast>(C, k)), expected);
}

TEST(random_greedy, small)
{
    const auto B     = gaussian(50, 20);
    const MatrixXd L = B.transpose() * B;
    const CachedGramMatrix C(B);
    const int k = 10;

    const auto expected = get_expected(L, {15, 1, 19, 17, 10, 5, 8, 11, 7, 14});

    EXPECT_THAT(sol_value(random_greedy<NonLazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<NonLazy, Fast>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Fast>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<NonLazy, Oracle>(C, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<NonLazy, Fast>(C, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Oracle>(C, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Fast>(C, k)), expected);
}

TEST(random_greedy, negative)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;

    const int k         = 1;
    const auto expected = get_expected(L, {7});

    EXPECT_THAT(sol_value(random_greedy<NonLazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<NonLazy, Fast>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(random_greedy<Lazy, Fast>(L, k)), expected);
}

TEST(stochastic_greedy, empty)
{
    const MatrixXd B = MatrixXd::Zero(3, 0);
    const MatrixXd L = B.transpose() * B;
    const CachedGramMatrix C(B);
    const int k = 0;

    const auto expected = get_expected(L, {});

    EXPECT_THAT(sol_value(stochastic_greedy<NonLazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<NonLazy, Fast>(L, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<Lazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<Lazy, Fast>(L, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<NonLazy, Oracle>(C, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<NonLazy, Fast>(C, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<Lazy, Oracle>(C, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<Lazy, Fast>(C, k)), expected);
}

TEST(stochastic_greedy, small)
{
    const auto B     = gaussian(50, 20);
    const MatrixXd L = B.transpose() * B;
    const CachedGramMatrix C(B);
    const int k = 10;

    const auto expected = get_expected(L, {10, 13, 15, 14, 18, 9, 4, 19, 3, 17});

    EXPECT_THAT(sol_value(stochastic_greedy<NonLazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<NonLazy, Fast>(L, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<Lazy, Oracle>(L, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<Lazy, Fast>(L, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<NonLazy, Oracle>(C, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<NonLazy, Fast>(C, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<Lazy, Oracle>(C, k)), expected);
    EXPECT_THAT(sol_value(stochastic_greedy<Lazy, Fast>(C, k)), expected);
}

TEST(stochastic_greedy, negative)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 0.1;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;

    const int k         = 4;
    const auto expected = get_expected(L, {});

    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Fast>(L, k).last()), expected);
}

TEST(interlace_greedy, empty)
{
    const MatrixXd B = MatrixXd::Zero(3, 0);
    const MatrixXd L = B.transpose() * B;
    const CachedGramMatrix C(B);
    const int k = 0;

    const auto expected = get_expected(L, {});

    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Oracle>(C, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Fast>(C, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Oracle>(C, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Fast>(C, k).last()), expected);
}

TEST(interlace_greedy, small)
{
    const auto B     = gaussian(50, 20);
    const MatrixXd L = B.transpose() * B;
    const CachedGramMatrix C(B);
    const int k = 10;

    const auto expected = get_expected(L, {13, 18, 7, 15, 17, 10, 5, 14, 11, 9});

    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Oracle>(C, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Fast>(C, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Oracle>(C, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Fast>(C, k).last()), expected);
}

TEST(interlace_greedy, linear)
{
    VectorXd v(8);
    v << 4.0, 3.0, 2.0, 0.5, 0.25, 0.125, 0.1, 0.1;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;

    const int k         = 4;
    const auto expected = get_expected(L, {0, 1});

    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Fast>(L, k).last()), expected);

    const auto result = interlace_greedy<NonLazy, Oracle>(L, k);
    EXPECT_THAT(result[0].solution, ElementsAre());
    EXPECT_THAT(result[1].solution, ElementsAre(0));
    EXPECT_THAT(result[2].solution, ElementsAre(0, 1));
    EXPECT_THAT(result[3].solution, ElementsAre(0, 1));
    EXPECT_THAT(result[4].solution, ElementsAre(0, 1));
}

TEST(interlace_greedy, full)
{
    VectorXd v(8);
    v << 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;

    const int k         = 8;
    const auto expected = get_expected(L, {7, 6, 4, 2, 0});

    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<NonLazy, Fast>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Oracle>(L, k).last()), expected);
    EXPECT_THAT(sol_value(interlace_greedy<Lazy, Fast>(L, k).last()), expected);
}

TEST(double_greedy, empty)
{
    const MatrixXd L = MatrixXd::Zero(0, 0);

    const auto expected = get_expected(L, {});

    EXPECT_THAT(sol_value(double_greedy<Oracle>(L)), expected);
    EXPECT_THAT(sol_value(double_greedy<Fast>(L)), expected);
}

TEST(double_greedy, linear)
{
    VectorXd v(8);
    v << 4.0, 3.0, 2.0, 0.5, 0.25, 0.125, 0.1, 0.1;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;

    const auto expected = get_expected(L, {0, 1, 2});

    EXPECT_THAT(sol_value(double_greedy<Oracle>(L)), expected);
    EXPECT_THAT(sol_value(double_greedy<Fast>(L)), expected);
}
