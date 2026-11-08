function(elml_add_test name)
  add_executable(${name} ${name}.c)
  target_link_libraries(${name} PRIVATE elml_train_lib)
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src)
  target_compile_options(${name} PRIVATE ${ELML_WARNINGS})
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT 120)
endfunction()

elml_add_test(test_arena)
elml_add_test(test_registry)
elml_add_test(test_format)
elml_add_test(test_forward)
elml_add_test(test_csv)
elml_add_test(test_ini)
elml_add_test(test_config)
elml_add_test(test_dataset)
elml_add_test(test_net)
elml_add_test(test_io)
elml_add_test(test_misc)
elml_add_test(test_kernels)

add_test(NAME test_cli
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test_cli.sh
          $<TARGET_FILE:elml_train>
          $<TARGET_FILE:elml_run>
          ${CMAKE_CURRENT_BINARY_DIR}/cli_work)
set_tests_properties(test_cli PROPERTIES TIMEOUT 180)
