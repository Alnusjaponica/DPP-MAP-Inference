cmake_minimum_required(VERSION 3.23)

set(CMAKE_TOOLCHAIN_FILE
    ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
    CACHE STRING "Vcpkg toolchain file"
)

project(lazydpp
    VERSION 1.0
    DESCRIPTION "The official implementation of \"Lazy and Fast Greedy MAP Inference for Determinantal Point Process\""
    HOMEPAGE_URL "https://github.com/Alnusjaponica/DPP-MAP-Inference"
    LANGUAGES CXX
)

find_package(Boost 1.78 REQUIRED program_options)
find_package(Eigen3 3.4 CONFIG REQUIRED)

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

add_library(${PROJECT_NAME}
    src/io.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra -Wpedantic -g -fdiagnostics-color $<$<BOOL:${ENABLE_OPENMP}>:-Xpreprocessor -fopenmp>)
target_link_libraries(${PROJECT_NAME} PUBLIC
    Boost::headers
    Boost::program_options
    Eigen3::Eigen
    $<$<BOOL:${ENABLE_OPENMP}>:OpenMP::OpenMP_CXX>
    ${OpenCV_LIBS}
)

# bin settings
add_executable(exp bin/exp.cpp)
target_link_libraries(exp ${PROJECT_NAME})

add_executable(double bin/double.cpp)
target_link_libraries(double ${PROJECT_NAME})

add_executable(gen_wishart bin/gen_wishart.cpp)
target_link_libraries(gen_wishart ${PROJECT_NAME})

add_executable(product bin/product.cpp)
target_link_libraries(product ${PROJECT_NAME})

add_executable(pre_process bin/pre_process.cpp)
target_link_libraries(pre_process ${PROJECT_NAME})

# test settings
enable_testing()
find_package(GTest 1.12 CONFIG REQUIRED)

add_executable(${PROJECT_NAME}-test
    test/cached_gram_matrix.cpp
    test/strategy.cpp
    test/algorithm.cpp
)
target_link_libraries(${PROJECT_NAME}-test
    ${PROJECT_NAME}
    GTest::gmock
    GTest::gmock_main
    GTest::gtest
    GTest::gtest_main
)

set_target_properties(${PROJECT_NAME}-test PROPERTIES OUTPUT_NAME runtest)
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}-test PROPERTIES TIMEOUT 6000)
