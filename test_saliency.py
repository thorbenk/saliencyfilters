import vigra
import _saliencyfilters
import numpy

img = vigra.impex.readImage("lena.jpg").view(numpy.ndarray)

sal, thresh =_saliencyfilters.saliency(img)
print "apative threshold: ", thresh
vigra.impex.writeImage(sal, "lena_sal.png")
