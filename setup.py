from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extra_compile_args = ['-O3','-ffast-math','-fopenmp']
extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        "pyFFTps.PowerSpec",  # the same name as your .pyx file
        sources=["pyFFTps/PowerSpec.pyx"],  # include your .c file here
        include_dirs=[numpy.get_include()],  # add numpy's include directory
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["m"],
    ),
    Extension(
        "pyFFTps.MAS_pyl",  # the same name as your .pyx file
        sources=["pyFFTps/MAS_pyl.pyx", "pyFFTps/MAS_c.c"],  # include your .c file here
        include_dirs=[numpy.get_include()],  # add numpy's include directory
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name='pyFFTps',
    version='1.0.0',
    author='Zhao Chen',
    author_email='chiyiru@sjtu.edu.cn',
    packages=find_packages(),  # automatically find all packages
    ext_modules=cythonize(extensions,
                          compiler_directives={'language_level' : "3"}),  
)
