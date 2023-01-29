# Lazy and Fast Greedy MAP Inference for Determinantal Point Process

This code is the official implementation of [Lazy and Fast Greedy MAP Inference for Determinantal Point Process](https://arxiv.org/abs/2206.05947).

## Requirements

- [CMake](https://cmake.org/) (version ≧ 3.23)
- [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/)
- [GNU Make](https://www.gnu.org/software/make/) or [Ninja](https://ninja-build.org/)
- C++ Compiler ([GNU Compiler Collection](https://gcc.gnu.org/) (GCC) / [Clang](https://clang.llvm.org/) / ...) compatible to C++17
  - GCC: version 7.1 or later
  - Clang: version 5.0 or later

## Compile

When first cloning this repository, run the following commands:

```sh
git submodule init
git submodule update
```

To compile C++ codes, run:

```sh
cmake --preset make  # replace "make" with "ninja" if you use Ninja
cmake --build --preset release
```

## Data Preprocessing

To generate the input data used in the experiment, follow these steps.
The resulting data will be stored to `data/`.

### Synthetic Datasets

To generate synthetic data, run the following:

```sh
./build/gen_wishart
```

### Real-world Datasets

To pre-process the real world datasets,
Please follow these steps:

#### MovieLens 25M

To get the primary data of MovieLens 25M dataset, run the following commands:

```sh
mkdir -p data
wget -P data https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip data/ml-25m.zip -d data
./build/pre_process -d movie_lens
```

#### Netflix Prize

To get Netflix Prize dataset, you need a Kaggle account.
Logging to Kaggle, download `archive.zip` from [here](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and store it to `data/`.
For pre-processing, run the following commands.

```sh
mkdir -p data
unzip data/archive.zip -d data/netflix_raw
./build/pre_process -d netflix
```

#### Computing Product Matrices

The matrix $L = B^\top B$ for Real-world datasets can be computed by the following (run on the root directory):

```sh
./build/product -d netflix
./build/product -d movie_lens
```

## Run Experiments

Run commands on the root directory.

### Greedy, RandomGreedy, StochasticGreedy, InterlaceGreedy

```sh
./build/exp -a [algorithm] -d [dataset_name] -m [input_matrix]
```

- `algorithm`: greedy (default), random, stochastic, interlace
- `dataset_name`: wishart, wishart_fixed_k, movie_lens, netflix
- `input_matrix`: B (default), L

### DoubleGreedy

```sh
./build/double -d [dataset_name]
```

- `dataset_name`: wishart, movie_lens, netflix

Experimental results will be stored to `result/` in the CSV format.

## License

The code is licensed MIT.
