set(TEST_NAMES
  test_distribution
  test_file_utils
  test_tally
  test_interpolate
  test_math
  test_mcpl_stat_sum
  test_mesh
  test_region
  test_interval
  # Add additional unit test files here
)

foreach(test ${TEST_NAMES})
  add_executable(${test} ${test}.cpp)
  target_link_libraries(${test} Catch2::Catch2WithMain libopenmc)
  add_test(NAME ${test} COMMAND ${test} WORKING_DIRECTORY ${UNIT_TEST_BIN_OUTPUT_DIR})
endforeach()

# Benchmark executable (not a test, just a utility)
add_executable(benchmark_volume benchmark_volume.cpp)
target_link_libraries(benchmark_volume libopenmc)
