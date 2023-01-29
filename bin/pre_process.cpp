#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

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

struct BinMatrix
{
    int num_items;
    int num_ratings;
    int num_nonzero_entries;
    vector<pair<int, int>> entries;
};

unordered_map<int, int> reindex_map(const vector<int>& id_vector)
{
    const set<int> id_set(id_vector.begin(), id_vector.end());
    unordered_map<int, int> id_map;
    int new_id = 0;
    for(const int old_id : id_set) {
        id_map[old_id] = new_id++;
    }
    return id_map;
}

pair<vector<int>, vector<int>> read_txt_netflix()
{
    vector<int> movie_ids, user_ids;
    int movie_id = -1;
    for(int i = 1; i <= 4; ++i) {
        const path fpath = path("data/netflix_raw/combined_data_" + to_string(i) + ".txt");
        ifstream fin(fpath);
        if(!fin) {
            cerr << "File not found: " << fpath << endl;
            exit(EXIT_FAILURE);
        }
        cout << "Loading " << fpath << "..." << flush;

        string row;
        while(fin >> row) {
            std::istringstream is_row(row);
            string val;
            getline(is_row, val, ',');
            if(val.back() == ':') {
                ++movie_id;
                continue;
            }
            const int user_id = stoi(val);
            getline(is_row, val, ',');
            const double rating = stod(val);
            if(rating < 4.0) {
                continue;
            }
            movie_ids.push_back(movie_id);
            user_ids.push_back(user_id);
        }
        cout << "finished." << endl;
    }
    return {movie_ids, user_ids};
}

pair<vector<int>, vector<int>> read_csv_movie_lens()
{
    const path fpath = path("data/ml-25m/ratings.csv");
    ifstream fin(fpath);
    if(!fin) {
        cerr << "File not found: " << fpath << endl;
        exit(EXIT_FAILURE);
    }
    cout << "Loading " << fpath << "..." << flush;

    // Discards header
    string header;
    fin >> header;

    string row;
    vector<int> movie_ids, user_ids;
    while(fin >> row) {
        std::istringstream is_row(row);
        string val;
        getline(is_row, val, ',');
        const int user_id = stoi(val);
        getline(is_row, val, ',');
        const int movie_id = stoi(val);
        getline(is_row, val, ',');
        const double rating = stod(val);
        if(rating < 4.0) {
            continue;
        }
        movie_ids.push_back(movie_id);
        user_ids.push_back(user_id);
    }
    cout << "finished." << endl;
    return {movie_ids, user_ids};
}

BinMatrix construct_01_matrix(const vector<int>& movie_ids, const vector<int>& user_ids)
{
    const auto movie_id_map = reindex_map(movie_ids);
    const auto user_id_map  = reindex_map(user_ids);

    const int num_items = movie_id_map.size(), num_ratings = user_id_map.size(),
              nnz = user_ids.size();

    vector<pair<int, int>> entries;
    for(int i = 0; i < nnz; ++i) {
        entries.emplace_back(movie_id_map.at(movie_ids[i]), user_id_map.at(user_ids[i]));
    }
    sort(entries.begin(), entries.end());

    return {num_items, num_ratings, nnz, entries};
}

void save_01_matrix(const BinMatrix& bmat, const path& fpath)
{
    create_directories(fpath.parent_path());
    ofstream fout(fpath);
    if(!fout) {
        cerr << "Could not open " << fpath << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Writing " << fpath << "..." << flush;

    const int d = bmat.num_items, n = bmat.num_ratings, nnz = bmat.num_nonzero_entries;
    fout << d << ' ' << n << ' ' << nnz << endl;

    for(const auto& entry : bmat.entries) {
        const auto [movie_id, user_id] = entry;
        fout << movie_id << ' ' << user_id << '\n';
    }
    cout << " finished." << endl;
}

int main(const int argc, const char* const* const argv)
{
    using namespace boost::program_options;

    string data;

    // clang-format off
    options_description desc("Options");
    desc.add_options()
        ("data,d", value(&data), "Dataset name to input. Possible options are: netflix, movie_lens")
        ("help,h", "Print this help message")
    ;
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
    // clang-format on

    if(data == "movie_lens") {
        const auto [movie_ids, user_ids] = read_csv_movie_lens();
        const auto bin_matrix            = construct_01_matrix(movie_ids, user_ids);
        save_01_matrix(bin_matrix, path("data") / data / "B.txt");
    }
    else if(data == "netflix") {
        const auto [movie_ids, user_ids] = read_txt_netflix();
        const auto bin_matrix            = construct_01_matrix(movie_ids, user_ids);
        save_01_matrix(bin_matrix, path("data") / data / "B.txt");
    }
    else {
        cerr << "Data name to input has to be netflix, or movie_lens" << endl;
        exit(EXIT_FAILURE);
    }
}
