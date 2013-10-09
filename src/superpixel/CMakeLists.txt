add_library( superpixel SHARED superpixel.cpp )
target_link_libraries( superpixel ${OpenCV_LIBS} )
