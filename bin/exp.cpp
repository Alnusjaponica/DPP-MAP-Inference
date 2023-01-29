#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <boost/program_options.hpp>

#include "algorithm/greedy.hpp"
#include "algorithm/interlace_greedy.hpp"
#include "algorithm/random_greedy.hpp"
#include "algorithm/stochastic_greedy.hpp"
#include "cached_gram_matrix.hpp"
#include "io.hpp"
#include "oracle.hpp"
#include "strategy.hpp"
#include "timer.hpp"

using namespace std;
using namespace filesystem;
using namespace Eigen;

template<class R, class M>
using Func = R (*)(M, int, const Param&);

// --------
// Selection for algorithms
// --------
struct Greedy
{
    static constexpr const char* NAME = "greedy";

    template<class S, class O, class M>
    static constexpr const Func<GreedyResult, M> func = greedy<S, O, M>;
};

struct RandomGreedy
{
    static constexpr const char* NAME = "random";

    template<class S, class O, class M>
    static constexpr const Func<Result, M> func = random_greedy<S, O, M>;
};

struct StochasticGreedy
{
    static constexpr const char* NAME = "stochastic";

    template<class S, class O, class M>
    static constexpr const Func<Result, M> func = stochastic_greedy<S, O, M>;
};

struct InterlaceGreedy
{
    static constexpr const char* NAME = "interlace";

    template<class S, class O, class M>
    static constexpr const Func<InterlaceResult, M> func = interlace_greedy<S, O, M>;
};

// --------
// I/O
// --------
ofstream open_csv(const path& fpath)
{
    cout << "Opening " << fpath << "... " << flush;

    create_directories(fpath.parent_path());
    const bool is_new = !exists(fpath);

    ofstream fout(fpath, ios_base::app);
    if(!fout) {
        cerr << "Cannot open " << absolute(fpath) << endl;
        exit(EXIT_FAILURE);
    }

    if(is_new) {
        fout
            << "seed,n,k,solution_size,time,value,computed_entries_L,oracle_calls,computed_offdiagonals_V"
            << endl;
        cout << "newly created." << endl;
    }
    else {
        cout << "finished." << endl;
    }

    return fout;
}

void write_result(
    ostream& out, const uint_fast32_t seed, const int n, const int k, const Result& result)
{
    out << setprecision(16) << seed << "," << n << "," << k << "," << result.solution.size() << ","
        << result.time << "," << result.value << "," << result.num_computed_entries_L << ","
        << result.num_oracle_calls << "," << result.num_computed_offdiagonals_V << endl;
}

// --------
// Selection for experiments
// --------
struct ChangeKContinuous
{
    static constexpr const double time_limit = 3600.0;

    template<class A, class S, class O, class M>
    static void run(const M& L, const path& fpath, const uint_fast32_t seed = 0)
    {
        auto fout = open_csv(fpath);
        const Timer timer;

        const auto result = A::template func<S, O, M>(L, L.cols(), {time_limit, seed, false});

        for(int i = 0; i < int(result.size()); ++i) {
            write_result(fout, seed, L.cols(), i, result[i]);
        }
    }
};

template<int step>
struct ChangeKDiscrete
{
    static constexpr const double time_limit = 3600.0;

    template<class A, class S, class O, class M>
    static void run(const M& L,
                    const path& fpath,
                    const uint_fast32_t seed = 0,
                    double time_limit        = numeric_limits<double>::infinity())
    {
        auto fout = open_csv(fpath);

        for(int k = 0; k <= L.cols() / 4; k += step) {
            cout << "Running k = " << k << "... " << flush;
            const auto result = A::template func<S, O, M>(L, k, {time_limit, seed, false});

            if(result.finished) {
                cout << "finished." << endl;
                write_result(fout, seed, L.cols(), k, result);
            }
            else {
                cout << "stopped." << endl;
            }

            if(result.last().time > time_limit) {
                break;
            }
        }
    }
};

struct ChangeN
{
    static constexpr const int k             = 200;
    static constexpr const double time_limit = 60.0;

    template<class A, class S, class O, class M>
    static void run(const vector<M>& Ls, const path& fpath, const uint_fast32_t seed = 0)
    {
        auto fout = open_csv(fpath);

        for(const auto& L : Ls) {
            const Timer timer;

            const auto result =
                A::template func<S, O, M>(L, k, {numeric_limits<double>::infinity(), seed, false});
            if(result.finished) {
                write_result(fout, seed, L.cols(), k, result.last());
            }

            if(result.last().time > time_limit) {
                break;
            }
        }
    }
};

// --------
// Branching
// --------
template<class E, class A, class D>
void branch(const string& data_name,
            const string& matrix_name,
            const D& data,
            const uint_fast32_t seed = 0)
{
    const path out = path("result") / A::NAME / data_name;

    E::template run<A, Lazy, Fast>(data, out / ("Lazy-Fast-" + matrix_name + ".csv"), seed);
    E::template run<A, NonLazy, Fast>(data, out / ("NonLazy-Fast-" + matrix_name + ".csv"), seed);
    E::template run<A, Lazy, Oracle>(data, out / ("Lazy-Oracle-" + matrix_name + ".csv"), seed);
    E::template run<A, NonLazy, Oracle>(
        data, out / ("NonLazy-Oracle-" + matrix_name + ".csv"), seed);
}

template<class E, class A>
void change_k_B(const string& data_name_input,
                const string& data_name_output,
                MatrixXd (*const load)(const path&),
                const uint_fast32_t seed = 0)
{
    cout << "Starting Changing k for " << A::NAME << " on " << data_name_output
         << " with B-input setting" << endl;

    const path input_dir = path("data") / data_name_input;
    const auto B         = load(input_dir / "B.txt");
    const CachedGramMatrix L(B);
    branch<E, A>(data_name_output, "B", L, seed);

    cout << endl;
}

template<class E, class A>
void change_k_L(const string& data_name_input,
                const string& data_name_output,
                const uint_fast32_t seed = 0)
{
    cout << "Starting Changing k for " << A::NAME << " on " << data_name_output
         << " with L-input setting" << endl;

    const path input_dir = path("data") / data_name_input;
    const auto L         = load_symmetric_matrix(input_dir / "L.txt");
    branch<E, A>(data_name_output, "L", L, seed);

    cout << endl;
}

template<class A>
void change_n_B(const string& data_name_input,
                const string& data_name_output,
                const uint_fast32_t seed = 0)
{
    cout << "Starting Fixed k for " << A::NAME << " on " << data_name_input
         << " with B-input setting" << endl;

    const path input_dir = path("data") / data_name_input;
    const int N          = 10;

    vector<MatrixXd> Bs(N);
    vector<CachedGramMatrix> Cs;
    for(int i = 0; i < N; ++i) {
        Bs[i] = load_matrix(input_dir / to_string((i + 1) * 1000) / "B.txt");
        Cs.emplace_back(Bs[i]);
    }
    branch<ChangeN, A>(data_name_output, "B-" + to_string(ChangeN::k), Cs, seed);

    cout << endl;
}

template<class A>
void change_n_L(const string& data_name_input,
                const string& data_name_output,
                const uint_fast32_t seed = 0)
{
    cout << "Starting Fixed k for " << A::NAME << " on " << data_name_input
         << " with B-input setting"
         << " with L-input setting" << endl;
    const path input_dir = path("data") / data_name_input;
    const int N          = 10;

    vector<MatrixXd> Ls;
    for(int i = 0; i < N; ++i) {
        Ls.push_back(load_symmetric_matrix(input_dir / to_string((i + 1) * 1000) / "L.txt"));
    }
    branch<ChangeN, A>(data_name_output, "L-" + to_string(ChangeN::k), Ls, seed);

    cout << endl;
}

int main(const int argc, const char* const* const argv)
{
    using namespace boost::program_options;

    string algorithm, data, matrix;

    // clang-format off
    options_description desc("Options");
    desc.add_options()
        ("algorithm,a", value(&algorithm)->default_value("greedy"), "Algorithm to run. Possible options are: greedy, random, stochastic, interlace")
        ("data,d", value(&data), "Data name to input. Possible options are: wishart, wishart_fixed_k, netflix, movie_lens")
        ("matrix,m", value(&matrix)->default_value("B"), "Input matrix type. Possible options are: B, L")
        ("help,h", "Print this help message")
    ;
    // clang-format on

    variables_map vm;

    try {
        store(parse_command_line(argc, argv, desc), vm);
    }
    catch(const error& e) {
        cerr << e.what() << endl;
        exit(EXIT_FAILURE);
    }

    notify(vm);

    if(vm.count("help") || data.empty()) {
        cout << desc << endl;
        exit(EXIT_SUCCESS);
    }

    if(matrix != "B" && matrix != "L") {
        cout << "Invalid matrix name: " << matrix << "\nChoose B or L." << endl;
        exit(EXIT_FAILURE);
    }

    // Load wishart using load_matrix. Other data is load by load_01_matrix.
    string data_input               = data;
    MatrixXd (*load_B)(const path&) = data == "wishart" ? load_matrix : load_01_matrix;
    if(data == "wishart") {
        data_input = "wishart/6000";
    }
    else if(data == "wishart_fixed_k") {
        data_input = "wishart";
    }

    const auto seed = random_device{}();

    if(algorithm == "greedy") {
        if(data == "wishart_fixed_k") {
            if(matrix == "B") {
                change_n_B<Greedy>(data_input, data, seed);
            }
            else {
                change_n_L<Greedy>(data_input, data, seed);
            }
        }
        else {
            if(matrix == "B") {
                change_k_B<ChangeKContinuous, Greedy>(data_input, data, load_B, seed);
            }
            else {
                change_k_L<ChangeKContinuous, Greedy>(data_input, data, seed);
            }
        }
    }
    else if(algorithm == "random") {
        if(data == "wishart_fixed_k") {
            if(matrix == "B") {
                change_n_B<RandomGreedy>(data_input, data, seed);
            }
            else {
                change_n_L<RandomGreedy>(data_input, data, seed);
            }
        }
        else {
            if(matrix == "B") {
                change_k_B<ChangeKDiscrete<200>, RandomGreedy>(data_input, data, load_B, seed);
            }
            else {
                change_k_L<ChangeKDiscrete<200>, RandomGreedy>(data_input, data, seed);
            }
        }
    }
    else if(algorithm == "stochastic") {
        if(data == "wishart_fixed_k") {
            if(matrix == "B") {
                change_n_B<StochasticGreedy>(data_input, data, seed);
            }
            else {
                change_n_L<StochasticGreedy>(data_input, data, seed);
            }
        }
        else {
            if(matrix == "B") {
                change_k_B<ChangeKDiscrete<200>, StochasticGreedy>(data_input, data, load_B, seed);
            }
            else {
                change_k_L<ChangeKDiscrete<200>, StochasticGreedy>(data_input, data, seed);
            }
        }
    }
    else if(algorithm == "interlace") {
        if(data == "wishart_fixed_k") {
            if(matrix == "B") {
                change_n_B<InterlaceGreedy>(data_input, data, seed);
            }
            else {
                change_n_L<InterlaceGreedy>(data_input, data, seed);
            }
        }
        else {
            if(matrix == "B") {
                change_k_B<ChangeKContinuous, InterlaceGreedy>(data_input, data, load_B, seed);
            }
            else {
                change_k_L<ChangeKContinuous, InterlaceGreedy>(data_input, data, seed);
            }
        }
    }
    else {
        cerr << "Unknown algorithm: " << algorithm << endl;
        exit(EXIT_FAILURE);
    }
}
