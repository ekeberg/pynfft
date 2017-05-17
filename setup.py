from distutils.core import setup, Extension
import numpy.distutils.misc_util 

ext = Extension("nfft", sources=["nfftmodule.c"], libraries=["nfft3"], include_dirs=[numpy.get_include(), "/usr/local/include"], library_dirs=["/usr/local/lib"])
setup(ext_modules=[ext], include_dirs=[numpy.get_include(), "/usr/local/include"])
