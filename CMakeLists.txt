cmake_minimum_required(VERSION 3.19)

project(CompressionSimulationStudy CXX C)
find_package(mgard REQUIRED)
find_package(ADIOS2 REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)

add_executable(cpu-application-simulator cpu-application-simulator.cpp)
target_link_libraries(cpu-application-simulator mgard::mgard adios2::adios2)