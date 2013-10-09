include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

if (TBB_FOUND)
	include_directories( ${TBB_INCLUDE_DIR} )
	add_definitions( -DUSE_TBB )
else(TBB_FOUND)
	set( TBB_LIBRARIES "" )
endif(TBB_FOUND)

add_executable( test_superpixel test_superpixel.cpp )
target_link_libraries( test_superpixel superpixel )

add_executable( test_saliency test_saliency.cpp )
target_link_libraries( test_saliency saliency )

add_executable( evaluate evaluate.cpp )
target_link_libraries( evaluate saliency ${TBB_LIBRARIES} )

add_subdirectory(filter)
add_subdirectory(saliency)
add_subdirectory(superpixel)
