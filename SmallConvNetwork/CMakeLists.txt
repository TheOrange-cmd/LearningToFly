cmake_minimum_required(VERSION 3.10)
project(ModelValidator C)



# Add executables
add_executable(model_runner_grid model_runner_grid.c model_rgb.c)

# Link math library only on non-Windows platforms
if(NOT WIN32)
    target_link_libraries(model_runner_grid m)
endif()