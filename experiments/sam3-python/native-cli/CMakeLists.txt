cmake_minimum_required(VERSION 3.24)
project(InlaySam3PanelCli LANGUAGES C CXX)

if(NOT DEFINED SAM3_SOURCE_DIR OR NOT EXISTS "${SAM3_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Set SAM3_SOURCE_DIR to the pinned Asher-1/sam3-ggml checkout.")
endif()

set(SAM3_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SAM3_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${SAM3_SOURCE_DIR}" sam3-runtime)

add_executable(sam3_panel_cli main.cpp)
target_link_libraries(sam3_panel_cli PRIVATE sam3)
target_compile_features(sam3_panel_cli PRIVATE cxx_std_14)
