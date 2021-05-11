cmake_minimum_required(VERSION 2.8)
project(assistant-service)

message(${CMAKE_SOURCE_DIR})

# Enable cout in .cc and .h files
if (${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
    add_definitions(-D_DEBUG)
endif (${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -fpermissive")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

# define third_party path for this project

# 设置环境目录
SET(CMAKE_THIRDPARTY_PATH "/home/yw.shi/env/tf241")

#指定运行时动态库的加载路径,ORIGIN指运行文件所在目录
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 
SET(CMAKE_INSTALL_RPATH

    "/home/yw.shi/env/tf241/lib/"
   )

# 设置构建目标
set(Main  "${CMAKE_SOURCE_DIR}/label_wav.cc")

# 设置头文件路径，所有依赖头文件都需添加进来
include_directories("/home/yw.shi/env/tf241/eigen3")
include_directories("/home/yw.shi/env/tf241")
include_directories("/home/yw.shi/env/tf241/compile")
include_directories("/home/yw.shi/env/tf241/third_party/abseil-cpp/include/")
include_directories("/home/yw.shi/env/tf241/protobuf/v3.9.2/include/")



# 设置动态库和静态库的路径，这里包括libtensorflow，absl静态库，protobuf动态库
link_directories("/home/yw.shi/env/tf241/lib")
link_directories("/home/yw.shi/env/tf241/third_party/abseil-cpp/lib")
link_directories("/home/yw.shi/env/tf241/protobuf/v3.9.2/lib/")


# 设置运行路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR})
add_executable(main ${Main})
# 建立链接
target_link_libraries(main tensorflow_cc tensorflow_framework)
