cmake_minimum_required(VERSION 3.18)

project(epic CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules (JEMALLOC jemalloc)

pkg_search_module(JEMALLOC REQUIRED jemalloc)
include_directories(${JEMALLOC_INCLUDE_DIRS})

add_definitions(-DCACHE_LINE_SIZE=64)

set(CUDA_FOUND ON)
if(DEFINED CUDA_FOUND)
    message(STATUS "cuda found")
else()
    message(STATUS "cuda not found")
endif()


# if (CUDA_FOUND)
#     set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#     enable_language(CUDA)
#     add_definitions(-DEPIC_CUDA_AVAILABLE)
#     set(CMAKE_CUDA_STANDARD 17)
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
#     set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# else ()
#     message(WARNING "CUDA not found, using CPU only. CUDA Related modules will not be available.")
# endif ()

message(STATUS "The value is: ${CUDA_FOUND}")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
enable_language(CUDA)
add_definitions(-DEPIC_CUDA_AVAILABLE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

set(EPIC_CPU_SOURCE
        util_log.cpp
        util_math.cpp
        cpu_auxiliary_range_index.cpp
        third_party/masstree/kvthread.cc
        third_party/masstree/string.cc
        third_party/masstree/straccum.cc
        benchmarks/ycsb.cpp
        benchmarks/tpcc_txn.cpp
        benchmarks/tpcc.cpp
        benchmarks/tpcc_index.cpp
        benchmarks/tpcc_txn_gen.cpp
        txn_bridge.cpp
        benchmarks/tpcc_cpu_executor.cpp
        benchmarks/ycsb_cpu_executor.cpp
)

set(GACCO_CPU_SOURCE
        gacco/benchmarks/tpcc.cpp
        gacco/benchmarks/tpcc_txn_gen.cpp
        gacco/benchmarks/ycsb.cpp)

set(EPIC_GPU_SOURCE
        gpu_allocator.cu
        gpu_execution_planner.cu
        benchmarks/gpu_ycsb.cu
        gpu_txn.cu
        util_gpu_transfer.cu
        benchmarks/tpcc_gpu_submitter.cu
        gpu_configuration.cu
        benchmarks/tpcc_gpu_executor.cu
        util_memory.cu
        benchmarks/tpcc_gpu_index.cu
        benchmarks/ycsb_gpu_index.cu
        benchmarks/ycsb_gpu_submitter.cu
        benchmarks/ycsb_gpu_executor.cu
        benchmarks/tpcc_gpu_aux_index.cu
        benchmarks/tpcc_gpu_txn.cu
        benchmarks/micro.cu
)

set(GACCO_GPU_SOURCE
    gacco/gpu_execution_planner.cu
        gacco/benchmarks/tpcc_gpu_submitter.cu
        gacco/benchmarks/tpcc_gpu_executor.cu
        gacco/benchmarks/ycsb_gpu_submitter.cu
        gacco/benchmarks/ycsb_gpu_executor.cu)

if (CUDA_FOUND)
    set(EPIC_SOURCE ${EPIC_CPU_SOURCE} ${EPIC_GPU_SOURCE} ${GACCO_CPU_SOURCE} ${GACCO_GPU_SOURCE})
else ()
    set(EPIC_SOURCE ${EPIC_CPU_SOURCE} ${GACCO_CPU_SOURCE})
endif ()

include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
        ${CMAKE_SOURCE_DIR}/third_party/masstree
        ${CMAKE_SOURCE_DIR}/third_party/gpubtree/include
)
if (CUDA_FOUND)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cuco/include)
endif()
add_library(epic STATIC ${EPIC_SOURCE})

# if (CUDA_FOUND)
#     set_property(TARGET epic PROPERTY CUDA_ARCHITECTURES 70)
# endif ()
set_property(TARGET epic PROPERTY CUDA_ARCHITECTURES 80)

target_link_libraries(epic ${JEMALLOC_LIBRARIES})

add_executable(epic_driver main.cpp)
set_property(TARGET epic_driver PROPERTY CUDA_ARCHITECTURES 80)
target_link_libraries(epic_driver epic)
target_link_libraries(epic_driver ${JEMALLOC_LIBRARIES})

add_executable(micro_driver micro_driver.cu)

set_property(TARGET micro_driver PROPERTY CUDA_ARCHITECTURES 80)

target_link_libraries(micro_driver epic)
target_link_libraries(micro_driver ${JEMALLOC_LIBRARIES})

add_subdirectory(tests)
