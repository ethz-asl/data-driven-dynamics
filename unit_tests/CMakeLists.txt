if(ENABLE_UNIT_TESTS OR CATKIN_ENABLE_TESTING)

# Uniform way to add unit tests. This works with catkin and with plain CMake.
function(add_unit_test target) # add sources as ARGN

    if(ENABLE_UNIT_TESTS)

        # Plain CMake to add google test
        add_executable(${target} ${ARGN})
        add_test(${target} ${target})

    elseif(CATKIN_ENABLE_TESTING)

        # Catkin google test facilities
        catkin_add_gtest(${target} ${ARGN})

    endif(ENABLE_UNIT_TESTS)

    # Link the gtest libraries in any case
    target_link_libraries(${target} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

endfunction(add_unit_test)


# Add the tests

    add_unit_test(gazebo_aerodynamics_plugin_test gazebo_aerodynamics_plugin_test.cpp)
    target_link_libraries(gazebo_aerodynamics_plugin_test gazebo_aerodynamics_plugin)


endif(ENABLE_UNIT_TESTS OR CATKIN_ENABLE_TESTING)
