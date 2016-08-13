 # sdm test CMakeLists.txt
cmake_minimum_required(VERSION 2.8.12)
project(sdm CXX)
set(serial "0.0.1")
set(soserial "1")
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
        return()
endif()

if (APPLE)
  message("${CMAKE_SYSTEM}")
elseif(UNIX)
  message("${CMAKE_SYSTEM}")
  find_package(OpenMP)
  if (OPENMP_FOUND)
      set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -ftree-vectorize -msse2  -mfpmath=sse")

set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}")


set( CommonInc
  #/usr/include/eigen3
  ${CMAKE_SOURCE_DIR}/../
  ${CMAKE_SOURCE_DIR}/../lib
  ${CMAKE_SOURCE_DIR}/../lib/eigen
)

include_directories( ${CONCERT_INC_DIR} ${CMAKE_SOURCE_DIR} ${CommonInc} )

set( CommonLibs
  m
  pthread
)

macro(subdirlsit result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
        list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

subdirlsit(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})

message("subdirectories : ${SUBDIRS}")

foreach(SUBDIR ${SUBDIRS})
  if (NOT ("CMakeFiles" STREQUAL "${SUBDIR}"))
    file(GLOB_RECURSE TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR} ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/*.cpp)
    message("files : ${TEST_SRCS}")
    foreach(SRC ${TEST_SRCS})
      get_filename_component(MY_EXE "${SRC}" NAME_WE)
      add_executable(${MY_EXE} ${SRC})
      target_link_libraries(${MY_EXE} ${CommonLibs})
    endforeach(SRC)
  endif()
endforeach(SUBDIR)
