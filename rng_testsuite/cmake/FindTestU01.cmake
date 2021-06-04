# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2021 Aalto University

set(TESTU01_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/TestU01-1.2.3/build CACHE PATH "Path to TestU01 install directory")
message(STATUS ${TESTU01_BASE_DIR})
find_path(TESTU01_INCLUDE_DIR NAMES TestU01.h HINTS ${TESTU01_BASE_DIR}/include)
find_library(TESTU01_LIBRARY testu01 libtestu01 HINTS ${TESTU01_BASE_DIR}/lib)
find_library(TESTU01_MYLIB_LIBRARY mylib libmylib HINTS ${TESTU01_BASE_DIR}/lib)
find_library(TESTU01_PROBDIST_LIBRARY probdist libprobdist HINTS ${TESTU01_BASE_DIR}/lib)

find_library(M_LIBRARY m REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TestU01 DEFAULT_MSG TESTU01_LIBRARY TESTU01_INCLUDE_DIR)

mark_as_advanced(TESTU01_INCLUDE_DIR TESTU01_LIBRARY)
set(TestU01_LIBRARIES ${TESTU01_LIBRARY} ${TESTU01_PROBDIST_LIBRARY} ${TESTU01_MYLIB_LIBRARY} ${M_LIBRARY})
set(TestU01_INCLUDE_DIRS ${TESTU01_INCLUDE_DIR})
