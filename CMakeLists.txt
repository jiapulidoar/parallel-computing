cmake_minimum_required(VERSION 2.8)
project( Reduction )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( reduction reduction.cpp)
add_executable( reduction_omp reduction_omp.cpp)

target_link_libraries( reduction ${OpenCV_LIBS}  -pthread -lm)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(reduction_omp PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS}  -pthread -lm)
endif()
