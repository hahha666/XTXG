cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(my_resnet_app)

# 寻找 LibTorch 库
find_package(Torch REQUIRED)

# 告诉 CMake 将你的 imp.cpp 编译成名为 imp 的可执行文件
add_executable(imp imp.cpp)

# 告诉 CMake 把 LibTorch 的依赖链接到你的程序上
target_link_libraries(imp "${TORCH_LIBRARIES}")
set_property(TARGET imp PROPERTY CXX_STANDARD 17)
set_property(TARGET imp PROPERTY CXX_STANDARD_REQUIRED ON)