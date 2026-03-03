cmake_minimum_required(VERSION 3.14)
project(tensorlearn)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(FetchContent)
FetchContent_Declare(
    lingebra
    GIT_REPOSITORY https://github.com/Lebogang-G-Masia/lingebra.git
    GIT_TAG        main 
)
FetchContent_Declare(
    unitgrad
    GIT_REPOSITORY https://github.com/Lebogang-G-Masia/unitgrad.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(lingebra unitgrad)
add_library(tensorlearn INTERFACE)
target_include_directories(tensorlearn INTERFACE include)
target_link_libraries(tensorlearn INTERFACE lingebra unitgrad)
add_executable(train src/main.cpp) 
target_link_libraries(train PRIVATE tensorlearn)
