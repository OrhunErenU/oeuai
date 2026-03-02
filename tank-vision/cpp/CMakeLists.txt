cmake_minimum_required(VERSION 3.18)
project(TankVision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
# Download from: https://github.com/microsoft/onnxruntime/releases
# Set ONNXRUNTIME_ROOT to the extracted directory
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime root directory")

if(NOT ONNXRUNTIME_ROOT)
    # Try common locations
    if(WIN32)
        set(_ort_search_paths
            "C:/onnxruntime"
            "C:/Program Files/onnxruntime"
            "$ENV{USERPROFILE}/onnxruntime"
        )
    else()
        set(_ort_search_paths
            "/usr/local"
            "/opt/onnxruntime"
            "$ENV{HOME}/onnxruntime"
        )
    endif()

    foreach(_path ${_ort_search_paths})
        if(EXISTS "${_path}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_ROOT "${_path}")
            break()
        endif()
    endforeach()
endif()

if(ONNXRUNTIME_ROOT)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_ROOT}")
else()
    message(FATAL_ERROR
        "ONNX Runtime not found. Download GPU release from:\n"
        "  https://github.com/microsoft/onnxruntime/releases\n"
        "Extract and set -DONNXRUNTIME_ROOT=<path>")
endif()

# ONNX Runtime include/lib
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_SOURCE_DIR}/include
)

find_library(ORT_LIB onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib)

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
    ${ORT_LIB}
)

# Copy ONNX Runtime DLL to output dir (Windows)
if(WIN32)
    file(GLOB ORT_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
    foreach(dll ${ORT_DLLS})
        add_custom_command(TARGET tank_vision POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}" $<TARGET_FILE_DIR:tank_vision>)
    endforeach()
endif()

install(TARGETS tank_vision DESTINATION bin)
