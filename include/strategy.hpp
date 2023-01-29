#ifndef STRATEGY_HPP
#define STRATEGY_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <queue>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utility.hpp"

// Non-lazy strategy
struct NonLazy
{
    template<class O>
    class Instance
    {
    private:
        O& oracle;
        std::unordered_set<int> T;
        bool add_dummy;
        std::vector<ElementValuePair> min_heap;

    public:
        // Initializes the ground set T to [begin, end)
        // If add_dummy is true, adds infinitely many dummy elements virtually
        template<class I>
        explicit Instance(O& oracle, const I begin, const I end, const bool add_dummy)
          : oracle(oracle), T(begin, end), add_dummy(add_dummy)
        {}

        // Returns a top element in T or std::nullopt if it is dummy
        std::optional<int> pop_largest()
        {
            return pop_kth_largest(0);
        }

        // Returns the ith top element in T or std::nullopt if it is dummy
        std::optional<int> pop_kth_largest(const int i)
        {
            assert(0 <= i && (add_dummy || i < int(T.size())));
            if(add_dummy && i >= int(T.size())) {
                return std::nullopt;
            }

            auto itr = T.begin();

            min_heap.resize(i + 1);
            for(int j = 0; j <= i; ++j) {
                min_heap[j] = {*itr, oracle.compute_marginal_gain_exponential(*itr)};
                ++itr;
            }

            std::make_heap(min_heap.begin(), min_heap.end(), std::greater());

            for(; itr != T.end(); ++itr) {
                const ElementValuePair next = {*itr,
                                               oracle.compute_marginal_gain_exponential(*itr)};
                if(min_heap[0] < next) {
                    std::pop_heap(min_heap.begin(), min_heap.end(), std::greater());
                    min_heap[i] = next;
                    std::push_heap(min_heap.begin(), min_heap.end(), std::greater());
                }
            }

            if(add_dummy && min_heap[0].value <= 1.0) {
                return std::nullopt;
            }
            else {
                remove(min_heap[0].element);
                return min_heap[0].element;
            }
        }

        // Removes e from T
        void remove(const int e)
        {
            assert(T.count(e) > 0);
            T.erase(e);
        }
    };

    NonLazy() = delete;

    template<class O, class I>
    static Instance<O> construct(O& oracle, const I begin, const I end, const bool add_dummy)
    {
        return Instance(oracle, begin, end, add_dummy);
    }
};

// Lazy strategy
struct Lazy
{
    template<class O>
    class Instance
    {
    private:
        O& oracle;
        std::unordered_set<int> T;
        bool add_dummy;
        std::priority_queue<ElementValuePair> Q;
        std::vector<int> recover;

    public:
        // Initializes the ground set T to [begin, end)
        // If add_dummy is true, adds infinitely many dummy elements virtually
        template<class I>
        explicit Instance(O& oracle, const I begin, const I end, const bool add_dummy)
          : oracle(oracle), T(begin, end), add_dummy(add_dummy)
        {
            std::vector<ElementValuePair> entries(T.size());
            transform(T.begin(), T.end(), entries.begin(), [&](const int e) {
                return ElementValuePair{e, oracle.get_last_marginal_gain_exponential(e)};
            });
            Q = std::priority_queue(entries.begin(), entries.end());
        }

        // Returns a top element in T or std::nullopt if it is dummy
        std::optional<int> pop_largest()
        {
            do {
                assert(add_dummy || !Q.empty());
                if(add_dummy && (Q.empty() || Q.top().value <= 1.0)) {
                    return std::nullopt;
                }

                auto [e, v] = Q.top();
                Q.pop();
                if(!T.count(e)) {
                    continue;
                }

                v = oracle.compute_marginal_gain_exponential(e);

                if(Q.empty() || v >= Q.top().value) {
                    if(add_dummy && v <= 1.0) {
                        return std::nullopt;
                    }
                    else {
                        remove(e);
                        return e;
                    }
                }

                Q.push({e, v});
            } while(1);
        }

        // Returns the ith top element in T or std::nullopt if it is dummy
        std::optional<int> pop_kth_largest(const int i)
        {
            assert(0 <= i && (add_dummy || i < int(T.size())));

            recover.clear();
            recover.reserve(i);

            for(int j = 0; j < i; ++j) {
                if(const auto e = pop_largest()) {
                    recover.push_back(*e);
                }
                else {
                    break;
                }
            }

            const auto ret = pop_largest();

            T.insert(recover.begin(), recover.end());
            for(const auto e : recover) {
                Q.push({e, oracle.get_last_marginal_gain_exponential(e)});
            }

            return ret;
        }

        // Removes e from T
        void remove(const int e)
        {
            assert(T.count(e) > 0);
            T.erase(e);
        }
    };

    Lazy() = delete;

    template<class O, class I>
    static Instance<O> construct(O& oracle, const I begin, const I end, const bool add_dummy)
    {
        return Instance(oracle, begin, end, add_dummy);
    }
};

#endif
