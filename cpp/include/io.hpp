#ifndef IO_HPP
#define IO_HPP

#include <cstdint>
#include <filesystem>

#include <Eigen/Core>

Eigen::MatrixXd load_01_matrix(const std::filesystem::path& fpath);

Eigen::MatrixXd load_matrix(const std::filesystem::path& fpath);

void save_matrix(const Eigen::MatrixXd& B,
                 const std::filesystem::path& fpath,
                 bool high_precision = false);

Eigen::MatrixXd load_symmetric_matrix(const std::filesystem::path& fpath);

void save_symmetric_matrix(const Eigen::MatrixXd& L,
                           const std::filesystem::path& fpath,
                           bool high_precision = false);

Eigen::MatrixXd gaussian(const int rows, const int cols, std::uint_fast32_t seed = 0);

Eigen::MatrixXd chen_matrix(const int rows, const int cols, std::uint_fast32_t seed = 0);

Eigen::MatrixXd distorted_gaussian(const int rows,
                                   const int cols,
                                   const int r,
                                   std::uint_fast32_t seed = 0);

Eigen::MatrixXd monotone_gaussian(const int rows, const int cols, std::uint_fast32_t seed = 0);

#endif
