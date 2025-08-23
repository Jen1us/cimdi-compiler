cmake_minimum_required(VERSION 3.20)
project(CimdiCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

include(AddLLVM)
include(AddMLIR)
include(TableGen)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(include)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(lib/Dialect/Cimdi)
add_subdirectory(lib/Conversion/LinalgToCimdi)
add_subdirectory(tools/cimdi-opt)