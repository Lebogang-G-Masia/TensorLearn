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

FetchContent_MakeAvailable(lingebra)

add_executable(train
    src/main.cpp 
    src/network.cpp
)

# 5. Link them together
target_link_libraries(train PRIVATE lingebra)
