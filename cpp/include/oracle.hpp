#ifndef ORACLE_HPP
#define ORACLE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>

// Oracle for log det L[S]
struct Oracle
{
    template<class M>
    class Instance
    {
    private:
        M& L;
        std::vector<int> u;
        std::vector<double> d;
        Eigen::LDLT<Eigen::MatrixXd> ldlt;
        std::vector<int> S;
        double value;
        int num_oracle_calls;

    public:
        explicit Instance(M& L, const int k)
          : L(L),
            u(L.cols(), -1),
            d(L.cols(), std::numeric_limits<double>::quiet_NaN()),
            ldlt(k),
            value(0.0),
            num_oracle_calls(0)
        {
            assert(0 <= k && k <= int(L.cols()));
            S.reserve(k);
        }

        // Computes the marginal gain of e w.r.t. the current solution
        double compute_marginal_gain_exponential(const int e)
        {
            assert(0 <= e && e < int(L.cols()));

            if(u[e] < int(S.size())) {
                ldlt.compute(L(S, S));

                // std::max cares numerical errors
                d[e] = std::max(0.0, L(e, e) - L(S, e).dot(ldlt.solve(L(S, e))));

                u[e] = S.size();
                ++num_oracle_calls;
            }

            return d[e];
        }

        // Returns the exponential of the last computed marginal gain of e
        double get_last_marginal_gain_exponential(const int e)
        {
            assert(0 <= e && e < int(L.cols()));
            if(u[e] == -1) {
                d[e] = L(e, e);
                u[e] = 0;
                ++num_oracle_calls;
            }
            return d[e];
        }

        // Adds e to S
        void add(const int e)
        {
            assert(0 <= e && e < int(L.cols()));
            S.push_back(e);
            value += log(d[e]);
        }

        // Returns the current solution S
        const std::vector<int>& get_solution() const
        {
            return S;
        }

        // Returns the current objective value
        double get_value() const
        {
            return value;
        }

        // Returns the number of oracle calls
        int get_num_oracle_calls() const
        {
            return num_oracle_calls;
        }

        // Returns the number of computed offdiagonals of V
        int get_num_computed_offdiagonals_V() const
        {
            return 0;
        }

        // Returns indices of computed offdiagonals of V
        std::vector<std::pair<int, int>> get_computed_offdiagonals_V() const
        {
            return {};
        }

        // Clears logs of computed indices of computed offdiagonals in V
        void clear_computed_offdiagonals_V() const {}
    };

    Oracle() = delete;

    template<class M>
    static Instance<M> construct(M& L, const int k, bool = false)
    {
        return Instance(L, k);
    }
};

// Fast Method
struct Fast
{
    template<class M>
    class Instance
    {
    private:
        M& L;
        std::vector<int> u;
        int U;
        std::vector<double> d;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V;
        std::vector<int> S;
        double value;

        bool log_computed_offdiagonals_V;
        std::vector<std::pair<int, int>> computed_offdiagonals_V;

    public:
        explicit Instance(M& L, const int k, const bool log_computed_offdiagonals_V)
          : L(L),
            u(L.cols(), 0),
            U(0),
            d(L.cols(), std::numeric_limits<double>::quiet_NaN()),
            V(L.cols(), k),
            value(0.0),
            log_computed_offdiagonals_V(log_computed_offdiagonals_V)
        {
            const int n = L.cols();
            assert(0 <= k && k <= n);

            S.reserve(k);

            if(log_computed_offdiagonals_V) {
                computed_offdiagonals_V.reserve(k * (k - 1) / 2 + k * (n - k));
            }
        }

        // Computes the marginal gain of e w.r.t. the current solution
        double compute_marginal_gain_exponential(const int e)
        {
            assert(0 <= e && e < int(L.cols()));
            get_last_marginal_gain_exponential(e);

            for(int j = u[e]; j < int(S.size()); j++) {
                const int l = S[j];
                V(e, j)     = (L(e, l) - V.row(e).head(j).dot(V.row(l).head(j))) / std::sqrt(d[l]);

                // std::max cares numerical errors
                d[e] = std::max(0.0, d[e] - V(e, j) * V(e, j));

                u[e]++;
                U++;

                if(log_computed_offdiagonals_V) {
                    computed_offdiagonals_V.emplace_back(e, l);
                }
            }

            return d[e];
        }

        // Computes the marginal gain of e w.r.t. the current solution
        double get_last_marginal_gain_exponential(const int e)
        {
            assert(0 <= e && e < int(L.cols()));

            if(std::isnan(d[e])) {
                d[e] = L(e, e);
            }

            return d[e];
        }

        // Adds e to S
        void add(const int e)
        {
            assert(0 <= e && e < int(L.cols()));
            S.push_back(e);
            value += std::log(d[e]);
        }

        // Returns the current solution S
        const std::vector<int>& get_solution() const
        {
            return S;
        }

        // Returns the current objective value
        double get_value() const
        {
            return value;
        }

        // Returns the number of oracle calls
        int get_num_oracle_calls() const
        {
            return 0;
        }

        // Returns the number of computed offdiagonals of V
        int get_num_computed_offdiagonals_V() const
        {
            return U;
        }

        // Returns the number of computed offdiagonals of V
        const std::vector<std::pair<int, int>>& get_computed_offdiagonals_V() const
        {
            return computed_offdiagonals_V;
        }

        // Clears logs of computed indices of computed offdiagonals in V
        void clear_computed_offdiagonals_V()
        {
            computed_offdiagonals_V.clear();
        }
    };

    Fast() = delete;

    template<class M>
    static Instance<M> construct(M& L, const int k, const bool log_computed_offdiagonals_V = false)
    {
        return Instance(L, k, log_computed_offdiagonals_V);
    }
};

#endif
