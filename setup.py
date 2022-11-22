import os
from setuptools import setup#, find_packages
from distutils.sysconfig import get_python_inc
#import pathlib

# python include dir
py_include_dir = os.path.join(get_python_inc())

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext

build_fpga=True
build_cuda=False

cwd = os.getcwd()
__version__ = "1.01"

ext_modules1 = [
    Pybind11Extension("HEAAN",
        ["bind/base64.cpp", "bind/heaan_wrapper.cpp"],
        # Example: passing in the version to the compiled code
        include_dirs=[py_include_dir,
         'pybind11/include',
          '/usr/local/include',
           'HEAAN/src'],
        language='c++',
        extra_compile_args=['-std=c++11'],
        extra_objects=['/usr/local/lib/libntl.so', 'HEAAN/lib/libHEAAN.a'],
        define_macros = [('VERSION_INFO', __version__)],
        package_dir = {'': 'hemul/'},
        )]
if build_fpga:
    ext_modules1.append(
        Pybind11Extension("HEAAN_fpga",
        ["bind/base64.cpp", "bind/heaan_wrapper_fpga.cpp"],
        # Example: passing in the version to the compiled code
        include_dirs=[py_include_dir,
         'pybind11/include',
          '/usr/local/include',
           'HEAAN_fpga/src', 'HEAAN_fpga/src_fpga'],
        language='c++',
        extra_compile_args=['-std=c++11'],
        extra_objects=['/usr/local/lib/libntl.so', 'HEAAN_fpga/lib/libHEAAN.a'],
        define_macros = [('VERSION_INFO', __version__)],
        package_dir = {'': 'hemul/'},
        )
    )

setup(
    name="hemul",
    version=__version__,
    author="DeepInsight",
    #packages=find_packages(),
    author_email="hschoi@dinsight.ai",
    url="",
    description="FHE Emulator",
    long_description="",
    ext_modules=ext_modules1,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    force=True # force recompile the shared library. 
)

import subprocess
proc = subprocess.Popen("mv *.cpython*.so ./hemul", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
