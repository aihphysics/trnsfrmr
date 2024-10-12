cmake_minimum_required( VERSION 3.20 )

project(transformer LANGUAGES CXX)
enable_testing()
include( CTest )
add_subdirectory( googletest )
include( GoogleTest )

add_compile_options( -O2 -std=c++23 -Wall -Wextra -pedantic -Werror ) 

add_executable(transformer src/main.cpp)

add_executable( gtestbatch "tests/tests.cpp" )
target_include_directories( gtestbatch PUBLIC "src" )
target_link_libraries( gtestbatch GTest::gtest_main )
gtest_discover_tests( gtestbatch )