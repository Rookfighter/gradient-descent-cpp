
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${CATCH2_INCLUDE_DIR}
)

set(TEST_SRC
    "main.cpp"

    "gradient_descent.cpp"
)

add_executable(run_tests ${TEST_SRC})
