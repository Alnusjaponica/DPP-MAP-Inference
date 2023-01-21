#ifndef CACHED_GRAM_MATRIX_HPP
#define CACHED_GRAM_MATRIX_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include <Eigen/Core>

// A class representing B.transpose() * B deferring entry calculations
class CachedGramMatrix
{
private:
    const Eigen::MatrixXd& B;
    Eigen::MatrixXd L;
    int num_computed_entries = 0;  // Ignore entries determined by the symmetricity

public:
    explicit CachedGramMatrix(const Eigen::MatrixXd& B)
      : B(B),
        L(Eigen::MatrixXd::Constant(B.cols(), B.cols(), std::numeric_limits<double>::quiet_NaN()))
    {}

    int rows() const
    {
        return L.rows();
    }

    int cols() const
    {
        return L.cols();
    }

    // Access to (i, j)th entry
    double operator()(const int i, const int j)
    {
        assert(0 <= i && i < rows() && 0 <= j && j < cols());
        if(std::isnan(L(i, j))) {
            L(i, j) = B.col(i).dot(B.col(j));
            L(j, i) = L(i, j);
            ++num_computed_entries;
        }
        return L(i, j);
    }

    auto col(const int j)
    {
        assert(0 <= j && j < cols());

        for(int i = 0; i < rows(); ++i) {
            (*this)(i, j);
        }
        return L.col(j);
    }

    auto operator()(const std::vector<int>& S, const int e)
    {
        assert(all_of(S.begin(), S.end(), [&](const int i) { return 0 <= i && i < rows(); })
               && 0 <= e && e < cols());

        for(const int i : S) {
            (*this)(i, e);
        }
        return L(S, e);
    }

    auto operator()(const std::vector<int>& S, const std::vector<int>& T)
    {
        assert(
            std::all_of(S.begin(), S.end(), [&](const int i) { return 0 <= i && i < rows(); })
            && std::all_of(T.begin(), T.end(), [&](const int j) { return 0 <= j && j < cols(); }));

        for(const int j : T) {
            (*this)(S, j);
        }
        return L(S, T);
    }

    friend int get_num_computed_entries(const CachedGramMatrix& L)
    {
        return L.num_computed_entries;
    }

    friend std::ostream& operator<<(std::ostream& out, const CachedGramMatrix& G)
    {
        return out << G.L << '\n';
    }
};

namespace Eigen
{

inline int get_num_computed_entries(const MatrixXd&)
{
    return 0;
}

}  // namespace Eigen

#endif