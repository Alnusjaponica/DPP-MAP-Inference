#include <filesystem>
#include <iostream>
#include <string>

#include <Eigen/Core>

#include "io.hpp"

using namespace std;
using namespace filesystem;
using namespace Eigen;

int main()
{
    for(int n = 1000; n <= 10000; n += 1000) {
        const path data_dir = path("data/wishart") / to_string(n);
        const auto B        = gaussian(n, n);
        save_matrix(B, data_dir / "B.txt", true);

        cout << "Computing L = B^T B... " << flush;
        const MatrixXd L = B.transpose() * B;
        cout << "finished." << endl;

        save_symmetric_matrix(L, data_dir / "L.txt", true);
    }
}
