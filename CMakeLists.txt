cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ctorch-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ctorch-app main.cpp)
target_link_libraries(ctorch-app "${TORCH_LIBRARIES}")
set_property(TARGET ctorch-app PROPERTY CXX_STANDARD 14)
