#include <filesystem>
#include <iostream>

#include <Eigen/Core>
#include <boost/program_options.hpp>

#include "io.hpp"
#include "timer.hpp"

using namespace std;
using namespace filesystem;
using namespace Eigen;

void run(const string& data_name)
{
    const path input_dir  = path("data") / data_name;
    const path output_dir = path("result/double") / data_name;

    const auto B = load_01_matrix(input_dir / "B.txt");
    cout << "Computing L = B^T B" << endl;
    const Timer timer;
    const MatrixXd L = B.transpose() * B;
    cout << "Time: " << timer.get() << endl;
    save_symmetric_matrix(L, input_dir / "L.txt");
    cout << endl;
}

int main(const int argc, const char* const* const argv)
{
    using namespace boost::program_options;

    string data;

    // clang-format off
    options_description desc("Options");
    desc.add_options()
        ("data,d", value(&data), "Data name to input. Possible options are: netflix, movie_lens")
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

    run(data);
}
