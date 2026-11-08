cmake_minimum_required(VERSION 3.25)

project(embedlml
  VERSION 0.1.0
  DESCRIPTION "Portable neural-network inference and training for embedded systems"
  LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ELML_TRAIN "Build the host-only trainer" ON)
option(ELML_BUILD_TESTS "Build the test suite" ON)
option(ELML_BUILD_EXAMPLES "Register the examples with CTest" ON)
option(ELML_SIMD "Use architecture-specific SIMD kernels when available" ON)
option(ELML_STRICT_FP "Force the scalar f32 path for reproducible results" OFF)
option(ELML_BUILD_STUDIO "Build the CUI training studio (downloads CUI if needed)" OFF)
set(ELML_CUI_SOURCE_DIR "" CACHE PATH "Local CUI checkout (optional)")
set(ELML_MAX_LAYERS "64" CACHE STRING "Maximum number of layers per model")
set(ELML_MAX_RANK "4" CACHE STRING "Maximum tensor rank")

set(ELML_WARNINGS
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wstrict-prototypes
  -Wmissing-prototypes
  -Wconversion
  -Wsign-conversion
  -Wcast-qual
  -Wundef
  -Wvla
  -Wdouble-promotion
  -Wformat=2
  -Werror)

set(ELML_SIMD_SOURCES)
set(ELML_SIMD_DEFINITIONS)
if(ELML_SIMD)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ELML_ARCH)
  if(ELML_ARCH MATCHES "^(x86_64|amd64)$")
    list(APPEND ELML_SIMD_SOURCES src/kernels/x86_64/simd_f32.c)
    list(APPEND ELML_SIMD_DEFINITIONS ELML_ARCH_X86_64=1)
  elseif(ELML_ARCH MATCHES "^(aarch64|arm64)$")
    list(APPEND ELML_SIMD_SOURCES src/kernels/arm64/neon_f32.c)
    list(APPEND ELML_SIMD_DEFINITIONS ELML_ARCH_ARM64=1)
  elseif(ELML_ARCH MATCHES "^(riscv64|riscv)$")
    list(APPEND ELML_SIMD_SOURCES src/kernels/riscv/vector_f32.c)
    list(APPEND ELML_SIMD_DEFINITIONS ELML_ARCH_RISCV=1)
  endif()
endif()

add_library(elml_core STATIC
  src/core/arena.c
  src/core/error.c
  src/core/registry.c
  src/core/model.c
  src/core/layers/dense.c
  src/core/layers/activation.c
  src/format/crc32.c
  src/format/read.c
  src/kernels/common/dispatch.c
  src/kernels/common/scalar_f32.c
  ${ELML_SIMD_SOURCES})

target_include_directories(elml_core
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_options(elml_core PRIVATE ${ELML_WARNINGS})
if(ELML_SIMD_DEFINITIONS)
  target_compile_definitions(elml_core PRIVATE ${ELML_SIMD_DEFINITIONS})
endif()
target_link_libraries(elml_core PUBLIC m)
target_compile_definitions(elml_core
  PUBLIC ELML_MAX_LAYERS=${ELML_MAX_LAYERS}
         ELML_MAX_RANK=${ELML_MAX_RANK})

if(ELML_STRICT_FP)
  target_compile_definitions(elml_core PUBLIC ELML_STRICT_FP=1)
endif()

add_executable(elml_inspect
  tools/inspect_main.c
  src/format/inspect.c)
target_link_libraries(elml_inspect PRIVATE elml_core)
target_include_directories(elml_inspect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(elml_inspect PRIVATE ${ELML_WARNINGS})

if(ELML_TRAIN)
  add_library(elml_train_lib STATIC
    src/format/write.c
    src/train/csv.c
    src/train/ini.c
    src/train/io.c
    src/train/config.c
    src/train/dataset.c
    src/train/net.c
    src/train/session.c
    src/train/export.c)

  target_link_libraries(elml_train_lib PUBLIC elml_core PRIVATE m)
  target_include_directories(elml_train_lib
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(elml_train_lib PRIVATE ${ELML_WARNINGS})

  add_executable(elml_train src/train/train_main.c)
  target_link_libraries(elml_train PRIVATE elml_train_lib)
  target_include_directories(elml_train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(elml_train PRIVATE ${ELML_WARNINGS})

  add_executable(elml_run src/run/run_main.c)
  target_link_libraries(elml_run PRIVATE elml_core elml_train_lib)
  target_include_directories(elml_run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(elml_run PRIVATE ${ELML_WARNINGS})
endif()

include(GNUInstallDirs)
install(TARGETS elml_core elml_inspect
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/elml DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(ELML_BUILD_TESTS AND ELML_TRAIN)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ELML_BUILD_EXAMPLES AND ELML_BUILD_TESTS AND ELML_TRAIN)
  enable_testing()
  add_test(NAME example_iris
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/examples/iris/run.sh
            ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/iris)
  set_tests_properties(example_iris PROPERTIES TIMEOUT 180)
  add_test(NAME example_wdbc
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/examples/wdbc/run.sh
            ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/wdbc)
  set_tests_properties(example_wdbc PROPERTIES TIMEOUT 180)
endif()

if(ELML_BUILD_STUDIO)
  if(NOT ELML_TRAIN)
    message(FATAL_ERROR "ELML_BUILD_STUDIO requires ELML_TRAIN=ON")
  endif()

  find_package(Threads REQUIRED)

  set(CUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  if(ELML_CUI_SOURCE_DIR)
    add_subdirectory(${ELML_CUI_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cui)
  else()
    include(FetchContent)
    FetchContent_Declare(cui
      GIT_REPOSITORY https://github.com/yappologistic/CUI.git
      GIT_TAG d0383c4d3c00b7769f6b11cefdf581e1800f2da3
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(cui)
  endif()

  add_executable(elml_studio src/studio/studio_main.c)
  target_link_libraries(elml_studio PRIVATE elml_train_lib cui::cui Threads::Threads m)
  target_include_directories(elml_studio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(elml_studio PRIVATE ${ELML_WARNINGS})
  target_compile_definitions(elml_studio PRIVATE
    ELML_STUDIO_EXAMPLE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/examples/iris/iris.ini")
endif()
