#ifndef LOGDET_HPP
#define LOGDET_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>

#include <Eigen/Cholesky>
#include <Eigen/Core>

struct ElementValuePair
{
    int element;
    double value;

    bool operator<(const ElementValuePair& rhs) const
    {
        return std::pair(value, -element) < std::pair(rhs.value, -rhs.element);
    }

    bool operator>(const ElementValuePair& rhs) const
    {
        return std::pair(value, -element) > std::pair(rhs.value, -rhs.element);
    }
};

struct OptionalElementValuePair
{
    std::optional<int> element;
    double value;
};

template<class M>
double logdet(const M& A)
{
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(A);

    double ret = 0.0;
    for(const double v : ldlt.vectorD()) {
        ret += std::log(v);
    }

    return ret;
}

template<class M>
std::optional<Eigen::MatrixXd> inverse(const M& A)
{
    assert(A.rows() == A.cols());
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(A);

    const auto d = ldlt.vectorD();
    if(std::find(d.begin(), d.end(), 0.0) != d.end()) {
        return std::nullopt;
    }

    const int n = A.rows();
    return ldlt.solve(Eigen::MatrixXd::Identity(n, n));
}

#endif
