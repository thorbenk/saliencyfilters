add_library( saliency SHARED saliency.cpp )
target_link_libraries( saliency superpixel filter )
