project(gpt_bpe)
cmake_minimum_required(VERSION 3.22)

set(TARGET_OUT test.out)
set(TARGET_LIB test.lib)

# Go configurations
set(GO_SRCS library.go)
set(GO_LIBNAME gpt_bpe)
set(GO_LIBFILE ${GO_LIBNAME}.dylib)

add_custom_command(OUTPUT
        ${CMAKE_CURRENT_SOURCE_DIR}/${GO_LIBFILE}
        ${CMAKE_CURRENT_SOURCE_DIR}/${GO_LIBNAME}.h
        DEPENDS ${GO_SRCS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND env go build -buildmode=c-shared
        -o "${CMAKE_CURRENT_SOURCE_DIR}/${GO_LIBFILE}"
        ${CMAKE_GO_FLAGS} ${GO_SRCS}
        COMMENT "Building Go library")

add_custom_command(OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/${GO_LIBFILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${GO_LIBFILE}
        COMMAND cp
        "${CMAKE_CURRENT_SOURCE_DIR}/${GO_LIBFILE}"
        "${CMAKE_CURRENT_BINARY_DIR}/${GO_LIBFILE}")

add_executable(gpt_bpe_test
        test/test.c
        library.h
        ${CMAKE_CURRENT_SOURCE_DIR}/${GO_LIBNAME}.h
        ${CMAKE_CURRENT_BINARY_DIR}/${GO_LIBFILE})
target_include_directories(gpt_bpe_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(gpt_bpe_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gpt_bpe_test
        ${CMAKE_CURRENT_SOURCE_DIR}/${GO_LIBFILE})
