#include "Python.h"                                                                                                     
#include <boost/python.hpp>                                                                                             
#include <vigra/numpy_array.hxx>                                                                                        
                                                                                                                        
#include "saliency/saliency.h"

boost::python::tuple
pySaliency(vigra::NumpyArray<3, float> img) {
    if(img.shape(2) != 3) {
        throw std::runtime_error("was expecting a RGB image");
    }
    Mat_<Vec3b> cvImg(img.shape(0), img.shape(1));
    for(int i=0; i<img.shape(0); ++i) {
        for(int j=0; j<img.shape(1); ++j) {
            for(int c=0; c<3; ++c) {
                cvImg(i,j)[c] = img(i,j,c);
            }
        }
    }
    
    Saliency saliency;
    Mat_<float> sal = saliency.saliency( cvImg );
    
    double adaptive_T = 2.0 * sum( sal )[0] / (sal.cols*sal.rows);
    while (sum( sal > adaptive_T )[0] == 0)
        adaptive_T /= 1.2;
    
    vigra::NumpyArray<2, float> SAL(vigra::Shape2(img.shape(0), img.shape(1)));
    for(int i=0; i<img.shape(0); ++i) {
        for(int j=0; j<img.shape(1); ++j) {
            SAL(i,j) = sal(i,j);
        }
    }
    return boost::python::make_tuple(SAL, adaptive_T);
}
                                                                                                                        
BOOST_PYTHON_MODULE(_saliencyfilters) {                                                                                            
    _import_array();                                                                                                    
    vigra::import_vigranumpy();                                                                                         
  
    boost::python::def("saliency", &pySaliency,
        (boost::python::arg("image")),
        "Compute the saliency of a given RGB image."
        "Returns a tuple (saliency, adaptiveThreshold)."
    );
}  