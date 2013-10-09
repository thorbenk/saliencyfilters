cmake_minimum_required (VERSION 2.6)
project(saliency)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package( OpenCV REQUIRED )
find_package( TBB )

find_package( PythonLibs REQUIRED )
find_package( Boost COMPONENTS python REQUIRED )
find_package( VIGRA REQUIRED )

add_definitions( -std=c++11 )

add_subdirectory (src)
