cmake_minimum_required( VERSION 3.20 )

project(transformer LANGUAGES CXX)

add_compile_options( -O2 -std=c++23 -Wall -Wextra -pedantic -Werror ) 

add_executable(transformer src/main.cxx)
