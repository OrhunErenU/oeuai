cmake_minimum_required(VERSION 3.18)
project(TankVision LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# TensorRT
set(TENSORRT_ROOT "C:/TensorRT" CACHE PATH "TensorRT root directory")
find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_ROOT}/lib)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_ROOT}/lib)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
    ${CMAKE_SOURCE_DIR}/include
)

# Tank Vision executable
add_executable(tank_vision
    src/main.cpp
    src/detector.cpp
    src/tracker.cpp
    src/hud.cpp
    src/threat.cpp
    src/depth.cpp
)

target_link_libraries(tank_vision
    ${OpenCV_LIBS}
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${CUDA_LIBRARIES}
)

# Install
install(TARGETS tank_vision DESTINATION bin)
