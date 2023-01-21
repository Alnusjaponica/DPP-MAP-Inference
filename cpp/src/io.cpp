#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include "boost/random/normal_distribution.hpp"

#include "io.hpp"

using namespace std;
using namespace filesystem;
using namespace Eigen;

MatrixXd load_01_matrix(const path& fpath)
{
    ifstream fin(fpath);
    if(!fin) {
        cerr << "File not found: " << fpath << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Loading " << fpath << "..." << flush;

    int n, d, nnz;
    fin >> n >> d >> nnz;

    MatrixXd B = MatrixXd::Zero(d, n);

    for(int i = 0; i < nnz; i++) {
        int m, u;
        fin >> m >> u;
        B(u, m) = 1.0;
    }

    cout << " finished." << endl;

    return B;
}

MatrixXd load_matrix(const path& fpath)
{
    ifstream fin(fpath);
    if(!fin) {
        cerr << "File not found: " << fpath << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Loading " << fpath << "..." << flush;

    int d, n;
    fin >> d >> n;

    MatrixXd B = MatrixXd::Zero(d, n);

    for(int i = 0; i < d; ++i) {
        for(int j = 0; j < n; ++j) {
            fin >> B(i, j);
        }
    }

    cout << " finished." << endl;

    return B;
}

void save_matrix(const Eigen::MatrixXd& B, const path& fpath, bool high_precision)
{
    create_directories(fpath.parent_path());

    ofstream fout(fpath);
    if(!fout) {
        cerr << "Could not open " << fpath << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Writing " << fpath << "..." << flush;

    const int d = B.rows(), n = B.cols();
    fout << d << ' ' << n << endl;

    if(high_precision) {
        fout << setprecision(16);
    }

    for(int i = 0; i < d; ++i) {
        for(int j = 0; j < n; ++j) {
            fout << (j > 0 ? " " : "") << B(i, j);
        }
        fout << '\n';
    }

    cout << " finished." << endl;
}

MatrixXd load_symmetric_matrix(const path& fpath)
{
    ifstream fin(fpath);
    if(!fin) {
        cerr << "File not found: " << fpath << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Loading " << fpath << "..." << flush;

    int n;
    fin >> n;

    MatrixXd L = MatrixXd::Zero(n, n);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j <= i; ++j) {
            fin >> L(i, j);
            L(j, i) = L(i, j);
        }
    }

    cout << " finished." << endl;

    return L;
}

void save_symmetric_matrix(const MatrixXd& L, const path& fpath, bool high_precision)
{
    assert(L.rows() == L.cols());
    create_directories(fpath.parent_path());

    ofstream fout(fpath);
    if(!fout) {
        cerr << "Could not open " << fpath << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Writing " << fpath << "..." << flush;

    const int n = L.cols();
    fout << n << endl;

    if(high_precision) {
        fout << setprecision(16);
    }

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j <= i; ++j) {
            fout << (j > 0 ? " " : "") << L(j, i);
        }
        fout << '\n';
    }

    cout << " finished." << endl;
}

MatrixXd gaussian(const int rows, const int cols, const uint_fast32_t seed)
{
    mt19937 engine(seed);
    boost::random::normal_distribution dist(0.0, 1.0);  // FlawFinder: ignore

    MatrixXd B(rows, cols);

    for(int j = 0; j < cols; ++j) {
        for(int i = 0; i < rows; ++i) {
            B(i, j) = dist(engine);
        }
    }

    return B;
}

Eigen::MatrixXd chen_matrix(const int rows, const int cols, std::uint_fast32_t seed)
{
    mt19937 engine(seed);
    boost::random::normal_distribution dist(0.0, 1.0);  // FlawFinder: ignore

    MatrixXd B = gaussian(rows, cols, seed);

    for(int j = 0; j < cols; ++j) {
        B.col(j).normalize();
        B.col(j) *= exp(0.01 * dist(engine) + 0.2);
    }

    return B;
}

MatrixXd distorted_gaussian(const int rows, const int cols, const int r, const uint_fast32_t seed)
{
    mt19937 engine(seed);
    boost::random::normal_distribution dist(0.0, 1.0);  // FlawFinder: ignore

    MatrixXd B(rows, cols);

    for(int j = 0; j < cols; ++j) {
        for(int i = 0; i < rows; ++i) {
            B(i, j) = dist(engine) + r;
        }
    }

    return B;
}

MatrixXd monotone_gaussian(const int rows, const int cols, const uint_fast32_t seed)
{
    mt19937 engine(seed);
    boost::random::normal_distribution dist(0.0, 1.0);  // FlawFinder: ignore

    MatrixXd B(rows, cols);

    for(int j = 0; j < cols; ++j) {
        for(int i = 0; i < rows; ++i) {
            B(i, j) = dist(engine) + 1;
        }
    }

    return B;
}
