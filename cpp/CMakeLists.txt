cmake_minimum_required(VERSION 3.13)
project(openwakeword_axera_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(AXERA_TARGET "AX650" CACHE STRING "AXERA target: AX650 or AX630C")
set_property(CACHE AXERA_TARGET PROPERTY STRINGS AX650 AX630C)
string(TOUPPER "${AXERA_TARGET}" AXERA_TARGET)
string(TOLOWER "${AXERA_TARGET}" AXERA_TARGET_LOWER)
if(AXERA_TARGET STREQUAL "AX650")
  set(KWS_EXECUTABLE openwakeword_ax650)
  set(DEFAULT_BSP_MSP_DIR
      "${CMAKE_SOURCE_DIR}/toolchains/ax650n_bsp_sdk/msp/out")
elseif(AXERA_TARGET STREQUAL "AX630C")
  set(KWS_EXECUTABLE openwakeword_ax630c)
  set(DEFAULT_BSP_MSP_DIR
      "${CMAKE_SOURCE_DIR}/toolchains/ax620e_bsp_sdk/msp/out/arm64_glibc")
else()
  message(FATAL_ERROR "AXERA_TARGET must be AX650 or AX630C")
endif()

set(BSP_MSP_DIR "${DEFAULT_BSP_MSP_DIR}" CACHE PATH
    "BSP directory containing include/ and lib/")
if(NOT EXISTS "${BSP_MSP_DIR}/include/ax_engine_api.h")
  message(FATAL_ERROR
          "${AXERA_TARGET} BSP not found: ${BSP_MSP_DIR}. "
          "Set BSP_MSP_DIR to the directory containing include/ and lib/.")
endif()

# 共享库（供二次开发链接）
add_library(openwakeword_ax SHARED
  src/openwakeword_ax.cpp
  src/engine_wrapper.cpp
)
target_include_directories(openwakeword_ax PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${BSP_MSP_DIR}/include
)
target_link_directories(openwakeword_ax PUBLIC ${BSP_MSP_DIR}/lib)
target_compile_options(openwakeword_ax PRIVATE
  -O3 -Wall -Wextra -Wpedantic
  -fPIC -fdata-sections -ffunction-sections
)
target_link_libraries(openwakeword_ax PRIVATE
  ax_sys ax_engine ax_interpreter pthread dl m
)
target_link_options(openwakeword_ax PRIVATE
  -Wl,--enable-new-dtags -Wl,--allow-shlib-undefined
)
set_target_properties(openwakeword_ax PROPERTIES
  VERSION 0.1.0
  SOVERSION 0
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

# 命令行示例（链接共享库）
add_executable(${KWS_EXECUTABLE}
  examples/main.cpp
)
target_include_directories(${KWS_EXECUTABLE} PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(${KWS_EXECUTABLE} PRIVATE
  AXERA_TARGET_NAME="${AXERA_TARGET}"
)
target_compile_options(${KWS_EXECUTABLE} PRIVATE
  -O3 -Wall -Wextra -Wpedantic
  -fdata-sections -ffunction-sections
)
target_link_libraries(${KWS_EXECUTABLE} PRIVATE
  openwakeword_ax ax_sys ax_engine ax_interpreter pthread dl m)
target_link_directories(${KWS_EXECUTABLE} PRIVATE ${BSP_MSP_DIR}/lib)
set_target_properties(${KWS_EXECUTABLE} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

# 产物目录：cpp/lib/<TARGET>/*.so, cpp/bin/<TARGET>/* 可执行文件
set_target_properties(openwakeword_ax PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib/${AXERA_TARGET_LOWER}"
)
set_target_properties(${KWS_EXECUTABLE} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/${AXERA_TARGET_LOWER}"
)
