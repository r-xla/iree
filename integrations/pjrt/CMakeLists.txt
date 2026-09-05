# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.26...3.29)

project(IREE_PJRT)
cmake_policy(SET CMP0069 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(IREE_CXX_STANDARD ${CMAKE_CXX_STANDARD})

enable_testing()

# This project uses explicit symbol visibility.
set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")

# All python binaries go into one tree.
set(IREE_PJRT_PYTHON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/python")
set(IREE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Enable LTO if supported.
option(IREE_PJRT_ENABLE_LTO "Enable LTO (link time optimization) if supported" OFF)
include(CheckIPOSupported)
check_ipo_supported(RESULT _ireert_lto_supported OUTPUT error)
if(IREE_PJRT_ENABLE_LTO)
  if(_ireert_lto_supported)
    message(STATUS "Enabling LTO")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "LTO not supported by toolchain bit requested (ignored)")
  endif()
endif()

# Customize defaults.
option(IREE_BUILD_COMPILER "Enable compiler for runtime-library build" ON)
option(IREE_BUILD_SAMPLES "Disable samples for runtime-library build" OFF)

# The plugin needs no LLVM, and with this it does not need one to configure
# either.
#
# It talks to the compiler through exactly two headers, embedding_api.h and
# loader.h, whose whole include closure is <stdbool.h>, <stddef.h>, <stdint.h>
# and api_support.h -- not one LLVM or MLIR header among them. The compiler
# proper is never linked: it is dlopened at run time from libIREECompiler.so
# (ireeCompilerLoadLibrary(), see common/dylib_platform.cc).
#
# Those two headers nonetheless live in iree::compiler::bindings::c::headers
# alongside mlir_interop.h, which does need MLIR, so that target depends
# unconditionally on IREELLVMIncludeSetup -- and the root CMakeLists creates
# that target only inside if(IREE_BUILD_COMPILER), after a full LLVM configure.
# Building the plugin would therefore drag in the bundled llvm-project
# submodule (~2GB, hours) purely to satisfy an include path that none of the
# headers we use needs.
#
# So under IREE_BUILD_COMPILER=OFF we supply a stub: an INTERFACE library that
# adds no include directories at all. Sound exactly while the plugin includes
# nothing needing LLVM; if that ever changes, the build breaks loudly at the
# missing header rather than silently, which is the failure mode we want.
#
# All of this has to happen *before* add_subdirectory() below: that is where
# compiler/bindings/c declares its dependency on the target, and also where
# iree_generate_export_targets() reads the export set (see below).
if(NOT IREE_BUILD_COMPILER AND NOT TARGET IREELLVMIncludeSetup)
  message(STATUS "IREE_BUILD_COMPILER=OFF: stubbing IREELLVMIncludeSetup")
  add_library(IREELLVMIncludeSetup INTERFACE)

  # iree_compiler_bindings_c_headers is exported in the "Compiler" set, and
  # CMake refuses to export a target whose dependencies are not themselves
  # exported -- so the stub has to join that set, the way the root CMakeLists
  # makes the real one join it via iree_install_targets().
  #
  # Registering it by hand rather than calling iree_install_targets() is
  # deliberate, and it is about order. That function is only defined once IREE
  # has been added, and by then it is too late: build_tools/cmake is processed
  # *inside* add_subdirectory(), and iree_generate_export_targets() reads
  # IREE_EXPORT_TARGETS_Compiler and calls export(TARGETS) right there --
  # then clears the property. Registering afterwards satisfies install(EXPORT),
  # which CMake resolves at generate time, but not the build-tree export(),
  # which has already run.
  #
  # No DESTINATION clauses: an INTERFACE library installs no artifacts, and
  # GNUInstallDirs has not been included at this point anyway.
  set_property(
    GLOBAL APPEND PROPERTY "IREE_EXPORT_TARGETS_Compiler" IREELLVMIncludeSetup
  )
  install(
    TARGETS IREELLVMIncludeSetup
    EXPORT IREEExported-Compiler
    COMPONENT IREEPublicLibraries-Compiler
    EXCLUDE_FROM_ALL
  )
endif()

# Include IREE.
message(STATUS "Including IREE from ${IREE_ROOT_DIR}")
add_subdirectory("${IREE_ROOT_DIR}" "iree_core" EXCLUDE_FROM_ALL)

# Include local sources.
# Handle various global definitions that need to be set at the global
# toolchain level.
iree_setup_toolchain()

# Setup protoc and protobuf library
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(protobuf_cc_library)

add_subdirectory(src)
add_subdirectory(third_party/pjrt_c_api)

#-------------------------------------------------------------------------------
# Convenience test target.
#-------------------------------------------------------------------------------

add_custom_target(iree-pjrt-test-deps)

set(IREE_PJRT_CTEST_ARGS "-L;^iree_pjrt.+")
add_custom_target(iree-pjrt-run-tests
  COMMENT
    "Run IREE PJRT Tests"
    WORKING_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}"
  USES_TERMINAL
  VERBATIM
  COMMAND_EXPAND_LISTS
  COMMAND
    "${CMAKE_COMMAND}" -E echo
    "The 'iree-pjrt-run-tests' target is a helper for running ctest. For advanced"
    "options, build dependencies and invoke ctest independently as in:"
  COMMAND
    "${CMAKE_COMMAND}" -E echo
    "  (cd ${CMAKE_CURRENT_BINARY_DIR} && cmake --build . --target iree-pjrt-run-tests && ctest ${IREE_PJRT_CTEST_ARGS} --output-on-failure)"
  COMMAND
    "${CMAKE_COMMAND}" -E echo
    "Run tests in parallel by setting a variable like CTEST_PARALLEL_LEVEL=25."
  COMMAND
    "${CMAKE_CTEST_COMMAND}" ${IREE_PJRT_CTEST_ARGS} --output-on-failure
)
add_dependencies(iree-pjrt-run-tests iree-pjrt-test-deps)
