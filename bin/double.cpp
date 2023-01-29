#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include <Eigen/Core>
#include <boost/program_options.hpp>

#include "algorithm/double_greedy.hpp"
#include "io.hpp"
#include "oracle.hpp"
#include "timer.hpp"
#include "utility.hpp"

using namespace std;
using namespace filesystem;
using namespace Eigen;

// --------
// I/O
// --------
ofstream open_csv(const path& fpath)
{
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
    }

    return fout;
}

void write_result(
    ostream& out, const uint_fast32_t seed, const int n, const int k, const Result& result)
{
    out << seed << "," << n << "," << k << "," << result.solution.size() << "," << result.time
        << "," << result.value << "," << result.num_computed_entries_L << ","
        << result.num_oracle_calls << "," << result.num_computed_offdiagonals_V << endl;
}

// --------
// Algorithm
// --------
template<class O>
void run(const MatrixXd& L,
         const MatrixXd& L_inv,
         const path& fpath,
         const uint_fast32_t seed = 0,
         double timelimit         = 86400.0)
{
    auto fout = open_csv(fpath);

    const Param param = {timelimit, seed};
    const auto result = double_greedy<O>(L, L_inv, param);

    if(result.finished) {
        write_result(fout, seed, L.cols(), L.cols(), result);
    }
    else {
        cout << "Time limit (" << param.time_limit << " sec) has exceeded." << endl;
    }
}

void experiment(const string& data_name, MatrixXd (*const load_B)(const path&), bool merge_identity)
{
    const path input_dir  = path("data") / data_name;
    const path output_dir = path("result/double") / data_name;

    MatrixXd L;
    if(exists(input_dir / "L.txt")) {
        L = load_symmetric_matrix(input_dir / "L.txt");
    }
    else {
        const auto B = load_B(input_dir / "B.txt");
        cout << "Computing L = B^T B" << endl;
        const Timer timer;
        L = B.transpose() * B;
        cout << "Time: " << timer.get() << endl;
        save_symmetric_matrix(L, input_dir / "L.txt");
    }

    MatrixXd X;
    if(merge_identity) {
        X = 0.9 * L + 0.1 * MatrixXd::Identity(L.rows(), L.cols());
    }
    else {
        X = L;
    }

    const string matrix_name = merge_identity ? "L_I" : "L";
    const path fpath         = input_dir / (matrix_name + "_inv.txt");
    MatrixXd X_inv;

    if(exists(fpath)) {
        X_inv = load_symmetric_matrix(fpath);
    }
    else {
        cout << "Computing inv(" << matrix_name << ")" << endl;
        const Timer timer;
        const auto ret = inverse(X);
        if(!ret) {
            cerr << "L is singular." << endl;
            exit(EXIT_FAILURE);
        }
        X_inv = *ret;
        cout << "Time: " << timer.get() << endl;
        save_symmetric_matrix(X_inv, fpath, true);
    }

    const auto seed = random_device{}();

    cout << "Running Fast" << endl;
    run<Fast>(X, X_inv, output_dir / "Fast.csv", seed);

    cout << "Running Oracle" << endl;
    run<Oracle>(X, X_inv, output_dir / "Oracle.csv", seed);
}

int main(const int argc, const char* const* const argv)
{
    using namespace boost::program_options;

    string data;

    // clang-format off
    options_description desc("Options");
    desc.add_options()
        ("data,d", value(&data), "Data name to input. Possible options are: 1000, 2000, ..., 10000, netflix, movie_lens")
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

    if(data == "netflix" || data == "movie_lens") {
        experiment(data, load_matrix, false);
    }
    else if(data == "wishart") {
        for(int n = 2000; n <= 10000; n += 2000) {
            if(n == 6000)
                continue;
            experiment("wishart/" + to_string(n), load_matrix, false);
        }
    }
    else {
        cerr << "Data name to input has to be 1000, 2000, ..., 10000, netflix, or movie_lens"
             << endl;
        exit(EXIT_FAILURE);
    }
}
