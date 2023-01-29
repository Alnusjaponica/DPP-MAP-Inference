#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "cached_gram_matrix.hpp"
#include "io.hpp"

using namespace std;
using namespace Eigen;
using namespace testing;

TEST(cached_gram_matrix, test)
{
    const auto B = gaussian(20, 10);
    CachedGramMatrix L(B);

    for(int i = 0; i < L.rows(); ++i) {
        for(int j = 0; j < L.cols(); ++j) {
            EXPECT_DOUBLE_EQ(L(i, j), B.col(i).dot(B.col(j)));
        }
    }
}
