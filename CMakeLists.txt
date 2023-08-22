# cmake needs this line
cmake_minimum_required(VERSION 3.22)

# Define project name
project(app LANGUAGES CXX CUDA)

# enable C++ 11
set (CMAKE_CXX_STANDARD 11)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# Weired missing packages need explicit find_package()
find_package(Eigen3 REQUIRED)
find_package(Iconv REQUIRED)
find_package(gflags REQUIRED)

# if not, use manual specification of the OpenCVConfig.cmake path
# and comment previous line of automatic finding
# find_package(OpenCV REQUIRED PATHS /usr/local/share/OpenCV NO_DEFAULT_PATH)  

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(app "")

# Sources
target_sources(app PRIVATE
    src/NvAnalysis.cu
    src/main.cpp
)

# Include
target_include_directories(app PUBLIC 
    include 
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link your application with OpenCV libraries
target_link_libraries(app PRIVATE ${OpenCV_LIBS} CUDA::cudart)

set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "72;87")
