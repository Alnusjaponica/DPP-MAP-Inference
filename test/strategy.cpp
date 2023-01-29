#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "io.hpp"
#include "oracle.hpp"
#include "strategy.hpp"

using namespace std;
using namespace Eigen;
using namespace testing;

TEST(nonlazy, pop_largest)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;
    auto oracle  = Oracle::construct(L, 8);

    const vector T = {0, 1, 2, 3, 4, 5, 6, 7};
    auto strategy  = NonLazy::construct(oracle, T.begin(), T.end(), false);

    ASSERT_EQ(strategy.pop_largest(), optional(7));
    ASSERT_EQ(strategy.pop_largest(), optional(2));
    ASSERT_EQ(strategy.pop_largest(), optional(3));
    ASSERT_EQ(strategy.pop_largest(), optional(0));
    ASSERT_EQ(strategy.pop_largest(), optional(4));
    ASSERT_EQ(strategy.pop_largest(), optional(1));
    ASSERT_EQ(strategy.pop_largest(), optional(5));
    ASSERT_EQ(strategy.pop_largest(), optional(6));
}

TEST(nonlazy, pop_kth_largest)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;
    auto oracle  = Oracle::construct(L, 8);

    const vector T = {0, 1, 2, 3, 4, 5, 6, 7};
    auto strategy  = NonLazy::construct(oracle, T.begin(), T.end(), false);

    ASSERT_EQ(strategy.pop_kth_largest(2), optional(3));
    ASSERT_EQ(strategy.pop_kth_largest(2), optional(0));
    ASSERT_EQ(strategy.pop_kth_largest(5), optional(6));
    ASSERT_EQ(strategy.pop_kth_largest(0), optional(7));
}

TEST(nonlazy, dummy_pop_largest)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;
    auto oracle  = Oracle::construct(L, 8);

    const vector T = {0, 1, 2, 3, 4, 5, 6, 7};
    auto strategy  = NonLazy::construct(oracle, T.begin(), T.end(), true);

    ASSERT_EQ(strategy.pop_largest(), optional(7));
    ASSERT_EQ(strategy.pop_largest(), nullopt);
    ASSERT_EQ(strategy.pop_largest(), nullopt);
    ASSERT_EQ(strategy.pop_largest(), nullopt);
}

TEST(nonlazy, dummy_pop_kth_largest)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;
    auto oracle  = Oracle::construct(L, 8);

    const vector T = {0, 1, 2, 3, 4, 5, 6, 7};
    auto strategy  = NonLazy::construct(oracle, T.begin(), T.end(), true);

    ASSERT_EQ(strategy.pop_kth_largest(1), nullopt);
    ASSERT_EQ(strategy.pop_kth_largest(2), nullopt);
    ASSERT_EQ(strategy.pop_kth_largest(0), optional(7));
    ASSERT_EQ(strategy.pop_kth_largest(0), nullopt);
}

TEST(lazy, pop_largest)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;
    auto oracle  = Oracle::construct(L, 8);

    const vector T = {0, 1, 2, 3, 4, 5, 6, 7};
    auto strategy  = Lazy::construct(oracle, T.begin(), T.end(), false);

    ASSERT_EQ(strategy.pop_largest(), optional(7));
    ASSERT_EQ(strategy.pop_largest(), optional(2));
    ASSERT_EQ(strategy.pop_largest(), optional(3));
    ASSERT_EQ(strategy.pop_largest(), optional(0));
    ASSERT_EQ(strategy.pop_largest(), optional(4));
    ASSERT_EQ(strategy.pop_largest(), optional(1));
    ASSERT_EQ(strategy.pop_largest(), optional(5));
    ASSERT_EQ(strategy.pop_largest(), optional(6));
}

TEST(lazy, pop_kth_largest)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;
    auto oracle  = Oracle::construct(L, 8);

    const vector T = {0, 1, 2, 3, 4, 5, 6, 7};
    auto strategy  = Lazy::construct(oracle, T.begin(), T.end(), false);

    ASSERT_EQ(strategy.pop_kth_largest(2), optional(3));
    ASSERT_EQ(strategy.pop_kth_largest(2), optional(0));
    ASSERT_EQ(strategy.pop_kth_largest(5), optional(6));
    ASSERT_EQ(strategy.pop_kth_largest(0), optional(7));
}

TEST(lazy, dummy_pop_largest)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;
    auto oracle  = Oracle::construct(L, 8);

    const vector T = {0, 1, 2, 3, 4, 5, 6, 7};
    auto strategy  = Lazy::construct(oracle, T.begin(), T.end(), true);

    ASSERT_EQ(strategy.pop_largest(), optional(7));
    ASSERT_EQ(strategy.pop_largest(), nullopt);
    ASSERT_EQ(strategy.pop_largest(), nullopt);
    ASSERT_EQ(strategy.pop_largest(), nullopt);
}

TEST(lazy, dummy_pop_kth_largest)
{
    VectorXd v(8);
    v << 0.3, 0.2, 0.9, 0.5, 0.25, 0.125, 0.1, 2.0;
    MatrixXd L   = MatrixXd::Zero(8, 8);
    L.diagonal() = v;
    auto oracle  = Oracle::construct(L, 8);

    const vector T = {0, 1, 2, 3, 4, 5, 6, 7};
    auto strategy  = Lazy::construct(oracle, T.begin(), T.end(), true);

    ASSERT_EQ(strategy.pop_kth_largest(1), nullopt);
    ASSERT_EQ(strategy.pop_kth_largest(2), nullopt);
    ASSERT_EQ(strategy.pop_kth_largest(0), optional(7));
    ASSERT_EQ(strategy.pop_kth_largest(0), nullopt);
}
