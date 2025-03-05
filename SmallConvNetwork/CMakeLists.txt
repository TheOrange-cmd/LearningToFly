cmake_minimum_required(VERSION 3.10)
project(ModelValidator C)



# Add executables
add_executable(model_runner model_runner.c light_object_detection.c)
add_executable(test_runner test_runner.c)
add_executable(file_test file_test.c)
add_executable(minimal_runner minimal_runner.c)
add_executable(heap_runner heap_runner.c)
add_executable(model_runner_heap model_runner_heap.c light_object_detection.c)
add_executable(model_runner_clean model_runner_clean.c light_object_detection.c)
add_executable(model_runner_grid model_runner_grid.c light_object_detection.c)

if(MSVC)
    target_compile_options(model_runner PRIVATE /Od /Zi)
    set_target_properties(model_runner PROPERTIES LINK_FLAGS "/DEBUG")
endif()

# Link math library only on non-Windows platforms
if(NOT WIN32)
    target_link_libraries(model_runner m)
endif()