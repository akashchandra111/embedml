cmake_minimum_required(VERSION 3.25)

# The version lives in the public header; parse it so CMake and the library
# cannot disagree.
set(ELML_VERSION_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/elml/version.h")
file(READ "${ELML_VERSION_HEADER}" ELML_VERSION_TEXT)
foreach(part MAJOR MINOR PATCH)
  string(REGEX MATCH "#define ELML_VERSION_${part}[ \t]+([0-9]+)"
         _version_match "${ELML_VERSION_TEXT}")
  if(NOT _version_match)
    message(FATAL_ERROR "${ELML_VERSION_HEADER}: missing ELML_VERSION_${part}")
  endif()
  set(ELML_VERSION_${part} "${CMAKE_MATCH_1}")
endforeach()

project(embedml
  VERSION ${ELML_VERSION_MAJOR}.${ELML_VERSION_MINOR}.${ELML_VERSION_PATCH}
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
option(ELML_BUILD_STUDIO "Build the CUI training studio (uses the CUI submodule)" OFF)
option(ELML_BUILD_WEB "Build the WebAssembly studio and run its Node smoke test" OFF)
option(ELML_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
set(ELML_WEB_JOBS "" CACHE STRING "Parallel compile jobs for the wasm build (default: nproc)")
set(ELML_CUI_SOURCE_DIR "" CACHE PATH "Local CUI checkout instead of the submodule (optional)")

set(ELML_SANITIZE_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
if(ELML_SANITIZE)
  add_compile_options(${ELML_SANITIZE_FLAGS})
  add_link_options(${ELML_SANITIZE_FLAGS})
endif()

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
    src/train/memfs.c
    src/train/config.c
    src/train/dataset.c
    src/train/net.c
    src/train/evaluate.c
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
  set(ELML_EXAMPLE_ENV "CC=${CMAKE_C_COMPILER}")
  if(ELML_SANITIZE)
    string(JOIN " " ELML_SANITIZE_FLAGS_TEXT ${ELML_SANITIZE_FLAGS})
    list(APPEND ELML_EXAMPLE_ENV
      "CFLAGS=-O1 -g ${ELML_SANITIZE_FLAGS_TEXT}"
      "LDFLAGS=${ELML_SANITIZE_FLAGS_TEXT}")
  endif()
  add_test(NAME example_iris
    COMMAND ${CMAKE_COMMAND} -E env ${ELML_EXAMPLE_ENV}
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/iris/run.sh
            ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/iris)
  set_tests_properties(example_iris PROPERTIES TIMEOUT 180)
  add_test(NAME example_wdbc
    COMMAND ${CMAKE_COMMAND} -E env ${ELML_EXAMPLE_ENV}
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/wdbc/run.sh
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
    set(ELML_CUI_DIR "${ELML_CUI_SOURCE_DIR}")
  else()
    set(ELML_CUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/CUI")
  endif()
  if(NOT EXISTS "${ELML_CUI_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "CUI sources not found at ${ELML_CUI_DIR}. Initialize the submodule "
      "with: git submodule update --init --recursive")
  endif()
  add_subdirectory("${ELML_CUI_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/cui")

  add_executable(elml_studio src/studio/studio_main.c)
  target_link_libraries(elml_studio PRIVATE elml_train_lib cui::cui Threads::Threads m)
  target_include_directories(elml_studio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(elml_studio PRIVATE ${ELML_WARNINGS})
  target_compile_definitions(elml_studio PRIVATE
    ELML_STUDIO_EXAMPLE_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/examples/iris/iris.ini")
endif()

if(ELML_BUILD_WEB)
  if(NOT ELML_TRAIN)
    message(FATAL_ERROR "ELML_BUILD_WEB requires ELML_TRAIN=ON")
  endif()

  # The browser build uses plain clang with the wasm32 target (no Emscripten)
  # and Node for the smoke test. Skip quietly when the tools are missing.
  find_program(ELML_WEB_CLANG NAMES clang)
  find_program(ELML_WEB_WASM_LD NAMES wasm-ld)
  find_program(ELML_WEB_LLVM_AR NAMES llvm-ar)
  find_program(ELML_WEB_NODE NAMES node)
  if(NOT ELML_WEB_CLANG OR NOT ELML_WEB_WASM_LD OR NOT ELML_WEB_LLVM_AR
     OR NOT ELML_WEB_NODE)
    message(WARNING
      "ELML_BUILD_WEB=ON but clang, wasm-ld, llvm-ar and node are not all "
      "available; the browser build is disabled.")
  else()
    set(ELML_WEB_OUT "${CMAKE_CURRENT_BINARY_DIR}/web")
    if(NOT ELML_WEB_JOBS)
      set(ELML_WEB_JOBS "${CMAKE_BUILD_PARALLEL_LEVEL}")
      if(NOT ELML_WEB_JOBS)
        cmake_host_system_information(RESULT ELML_WEB_JOBS QUERY NUMBER_OF_LOGICAL_CORES)
      endif()
    endif()

    set(ELML_WEB_ENV
      "CC=${ELML_WEB_CLANG}"
      "WASM_LD=${ELML_WEB_WASM_LD}"
      "LLVM_AR=${ELML_WEB_LLVM_AR}"
      "ELML_WEB_OUT=${ELML_WEB_OUT}"
      "JOBS=${ELML_WEB_JOBS}")

    add_custom_target(elml_web ALL
      COMMAND ${CMAKE_COMMAND} -E env ${ELML_WEB_ENV}
              ${CMAKE_CURRENT_SOURCE_DIR}/wasm/build.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Building the WebAssembly studio (build/web)")

    enable_testing()
    add_test(NAME wasm_studio
      COMMAND ${CMAKE_COMMAND} -E env ${ELML_WEB_ENV}
              ${CMAKE_CURRENT_SOURCE_DIR}/wasm/ctest.sh "${ELML_WEB_NODE}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(wasm_studio PROPERTIES TIMEOUT 600)
  endif()
endif()
