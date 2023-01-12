# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2023 Aalto University

cmake_minimum_required(VERSION 3.21)
project(jax-chacha20-prng LANGUAGES CXX)

option(BUILD_TESTING "Build tests for native kernels" OFF)
option(FORCE_GENERIC "Build without CPU architecture optimized instructions" OFF)
set(HIP_ARCHITECTURES "gfx900 gfx906 gfx908 gfx90a gfx940 gfx1010 gfx1030 gfx1100" CACHE STRING "A list of AMD GPU architectures to built kernels for")

set(CMAKE_CXX_STANDARD 17)
set(cpu_arch_path "${CMAKE_CURRENT_LIST_DIR}/lib/generic/")
set(cpu_arch_def "ARCH_GENERIC")
set(SSE_ENABLED "No")
set(NEON_ENABLED "No")

if(NOT FORCE_GENERIC)
    try_run(SSE_RUN_SUCCESS SSE_COMPILE_SUCCESS
        ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config/sse_test.cpp
        #WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cmake_config/
    )

    if (NOT SSE_COMPILE_SUCCESS)
        # could not compile sse intrinsics, maybe we need the compiler flag?
        try_run(SSE_RUN_SUCCESS SSE_COMPILE_SUCCESS
            ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config/sse_test.cpp
            COMPILE_DEFINITIONS "-msse"
        )
        if (SSE_COMPILE_SUCCESS)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-msse>)
        endif()
    endif()

    if (NOT SSE_COMPILE_SUCCESS)
        try_run(NEON_RUN_SUCCESS NEON_COMPILE_SUCCESS
            ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config/neon_test.cpp
        )

        if (NOT NEON_COMPILE_SUCCESS)
            # could not compile neon intrinsics, maybe we need the compiler flag?
            try_run(NEON_RUN_SUCCESS NEON_COMPILE_SUCCESS
                ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_config/neon_test.cpp
                COMPILE_DEFINITIONS "-mfpu=neon"
            )
            if (NEON_COMPILE_SUCCESS)
                add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mfpu=neon>)
            endif()
        endif()
    endif()

    if (SSE_COMPILE_SUCCESS)
        set(cpu_arch_path "${CMAKE_CURRENT_LIST_DIR}/lib/intel/")
        set(cpu_arch_def "ARCH_INTEL")
        set(SSE_ENABLED "Yes")
        if (NOT SSE_RUN_SUCCESS EQUAL 0)
            message(WARNING "Can compile with Intel SSE instructions but failed a test run (cross-compiling?); will compile with SSE anyways - set FORCE_GENERIC to disable")
        endif()
    endif()

    if (NEON_COMPILE_SUCCESS)
        set(cpu_arch_path "${CMAKE_CURRENT_LIST_DIR}/lib/arm/")
        set(cpu_arch_def "ARCH_ARM")
        set(NEON_ENABLED "Yes")
        if (NOT NEON_RUN_SUCCESS EQUAL 0)
            message(WARNING "Can compile with ARM Neon instructions but failed a test run (cross-compiling?); will compile with Neon anyways - set FORCE_GENERIC to disable")
        endif()
    endif()

endif()


message("-- Detected architecture - " ${CMAKE_SYSTEM_PROCESSOR})

message("-- SSE instructions enabled - ${SSE_ENABLED}")
message("-- ARM Neon instructions enabled - ${NEON_ENABLED}")

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

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(extern/pybind11)
# find_package(pybind11 REQUIRED) # broken; does not use correct Python interpreter/libraries in a conda env

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_compile_definitions(OPENMP_AVAILABLE)
else()
    message(WARNING "OpenMP not found - Compiling without, but you may see lower performance.")
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

pybind11_add_module(
    native
    ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/lib/python_bindings.cpp)
target_include_directories(native PRIVATE ${cpu_arch_path})
target_compile_definitions(native PRIVATE ${cpu_arch_def})

if(OpenMP_CXX_FOUND)
    target_link_libraries(native PUBLIC OpenMP::OpenMP_CXX)
endif()

if (BUILD_TESTING)
    add_executable(cpu_kernel_tests
        ${CMAKE_CURRENT_LIST_DIR}/tests/lib/cpu_kernel_tests.cpp
        ${CMAKE_CURRENT_LIST_DIR}/lib/cpu_kernel.cpp)
    target_include_directories(cpu_kernel_tests PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/lib/
        ${cpu_arch_path})
    target_compile_definitions(cpu_kernel_tests PRIVATE ${cpu_arch_def})
    if (OpenMP_CXX_FOUND)
        target_link_libraries(cpu_kernel_tests PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_compile_definitions(native PRIVATE CUDA_ENABLED)
    target_sources(native PRIVATE ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_kernel.gpu.cpp)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/lib/gpu_kernel.gpu.cpp PROPERTIES LANGUAGE CUDA)
    # compile for lowest compute capability supported by XLA: 3.5
    # cf. https://www.tensorflow.org/install/gpu#hardware_requirements
    set_property(TARGET native PROPERTY CUDA_ARCHITECTURES 35)

    if (BUILD_TESTING)
        add_executable(gpu_kernel_tests
            ${CMAKE_CURRENT_LIST_DIR}/tests/lib/gpu_kernel_tests.gpu.cpp
            ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_kernel.gpu.cpp)
        set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/tests/lib/gpu_kernel_tests.gpu.cpp PROPERTIES LANGUAGE CUDA)
        target_compile_definitions(gpu_kernel_tests PRIVATE CUDA_ENABLED)
        target_include_directories(gpu_kernel_tests PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/lib/)
        target_compile_definitions(gpu_kernel_tests PRIVATE ${cpu_arch_def})
        set_property(TARGET gpu_kernel_tests PROPERTY CUDA_ARCHITECTURES 35)
        if (OpenMP_CXX_FOUND)
            target_link_libraries(gpu_kernel_tests PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()
else()
    message(WARNING "CUDA not found - not building CUDA GPU kernels!")
endif()

check_language(HIP)
separate_arguments(HIP_ARCHITECTURE_LIST UNIX_COMMAND ${HIP_ARCHITECTURES})

if (CMAKE_HIP_COMPILER)
    enable_language(HIP)

    target_compile_definitions(native PRIVATE HIP_ENABLED)
    target_sources(native PRIVATE ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_kernel.gpu.cpp)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/lib/gpu_kernel.gpu.cpp PROPERTIES LANGUAGE HIP)
    set_property(TARGET native PROPERTY HIP_ARCHITECTURES ${HIP_ARCHITECTURE_LIST})


    if (BUILD_TESTING)
        add_executable(gpu_kernel_tests
            ${CMAKE_CURRENT_LIST_DIR}/tests/lib/gpu_kernel_tests.gpu.cpp
            ${CMAKE_CURRENT_LIST_DIR}/lib/gpu_kernel.gpu.cpp)
        set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/tests/lib/gpu_kernel_tests.gpu.cpp PROPERTIES LANGUAGE HIP)
        target_compile_definitions(gpu_kernel_tests PRIVATE HIP_ENABLED)
        target_include_directories(gpu_kernel_tests PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/lib/)
        target_compile_definitions(gpu_kernel_tests PRIVATE ${cpu_arch_def})
        set_property(TARGET gpu_kernel_tests PROPERTY HIP_ARCHITECTURES ${HIP_ARCHITECTURE_LIST})
        if (OpenMP_CXX_FOUND)
            target_link_libraries(gpu_kernel_tests PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()
else()
    message(WARNING "HIP not found - not building AMD GPU kernels!")
endif()

install(TARGETS native DESTINATION chacha)
