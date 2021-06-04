# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

cmake_minimum_required(VERSION 3.12.0)
project(rng_tests VERSION 0.1.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(TestU01 REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Development)
message(STATUS ${TestU01_INCLUDE_DIRS})
message(STATUS ${TestU01_LIBRARIES})
message(STATUS ${Python3_LIBRARIES})
message(STATUS ${Python3_INCLUDE_DIRS})

add_library(libcchacha chacha_reference/chacha.c chacha_reference/chacharng.cpp chacha_reference/chacharng.hpp)
target_include_directories(libcchacha PUBLIC ${TestU01_INCLUDE_DIRS})

add_executable(rng_tests_chacha_c c_chacha_main.cpp)
target_compile_features(rng_tests_chacha_c PUBLIC cxx_auto_type cxx_deleted_functions cxx_inheriting_constructors cxx_generalized_initializers cxx_range_for)
target_include_directories(rng_tests_chacha_c PUBLIC ${TestU01_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/extern/argparse)
target_link_libraries(rng_tests_chacha_c PUBLIC ${TestU01_LIBRARIES} libcchacha)

add_executable(rng_tests main.cpp python_wrapper.cpp python_wrapper.hpp python_rng.cpp python_rng.hpp testu01_wrapper.cpp testu01_wrapper.hpp)
target_compile_features(rng_tests PUBLIC cxx_auto_type cxx_deleted_functions cxx_inheriting_constructors cxx_generalized_initializers cxx_range_for)
target_include_directories(rng_tests PUBLIC ${TestU01_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/extern/argparse)
target_link_libraries(rng_tests PUBLIC ${TestU01_LIBRARIES} ${Python3_LIBRARIES})
