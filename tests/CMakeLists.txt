set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

file(GLOB_RECURSE CPP_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CUDA_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

foreach(TEST_SRC ${CPP_TEST_SOURCES})

    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SRC})

    target_link_libraries(${TEST_NAME} GTest::GTest GTest::Main Threads::Threads ${PROJECT_NAME} spdlog::spdlog)

endforeach()

foreach(TEST_SRC ${CUDA_TEST_SOURCES})

    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SRC})

    target_link_libraries(${TEST_NAME} GTest::GTest GTest::Main Threads::Threads ${PROJECT_NAME} spdlog::spdlog)

endforeach()
