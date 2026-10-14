# CMake generated Testfile for 
# Source directory: /home/afa/CLionProjects/AE_Arch_Engine
# Build directory: /home/afa/CLionProjects/AE_Arch_Engine/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("AE_CppCoreTests" "/home/afa/CLionProjects/AE_Arch_Engine/build/ae_tests")
set_tests_properties("AE_CppCoreTests" PROPERTIES  _BACKTRACE_TRIPLES "/home/afa/CLionProjects/AE_Arch_Engine/CMakeLists.txt;70;add_test;/home/afa/CLionProjects/AE_Arch_Engine/CMakeLists.txt;0;")
add_test("AE_PythonBindingsTests" "/usr/bin/python3.14" "/home/afa/CLionProjects/AE_Arch_Engine/tests/test_bindings.py")
set_tests_properties("AE_PythonBindingsTests" PROPERTIES  _BACKTRACE_TRIPLES "/home/afa/CLionProjects/AE_Arch_Engine/CMakeLists.txt;74;add_test;/home/afa/CLionProjects/AE_Arch_Engine/CMakeLists.txt;0;")
