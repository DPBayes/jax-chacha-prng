# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

cmake_minimum_required(VERSION 3.18)
project(jax-chacha20-prng LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

# native kernels require pybind11; however, if it is installed from pip, it has issues
# in detecting the Python interpreter and libraries during CMake build in a conda env,
# despite what the docs say. We instead check out pybind11 as a submodule.

# initialize/update pybind11 submodule
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE_UPDATE "Check submodules during build" ON)
    if(GIT_SUBMODULE_UPDATE)
        message(STATUS "Initializing/updating pybind11 dependency...")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}.")
        endif()
    else()
        message(STATUS "Not updating submodules since GIT_SUBMODULE_UPDATE was OFF.")
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    message(FATAL_ERROR "The pybind11 repository was not downloaded! Please run manually: git submodule update --init --recursive .")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(extern/pybind11)
# find_package(pybind11 REQUIRED) # broken; does not use correct Python interpreter/libraries in a conda env

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

pybind11_add_module(
    native
    ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_kernel.cpp.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/python_bindings.cpp)
set_property(TARGET native PROPERTY CUDA_ARCHITECTURES 52)
install(TARGETS native DESTINATION chacha)
