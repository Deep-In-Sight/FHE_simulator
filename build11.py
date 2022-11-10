import os
from distutils.sysconfig import get_python_inc
from pybind11.setup_helpers import Pybind11Extension, build_ext

# python include dir
py_include_dir = os.path.join(get_python_inc())

def build(setup_kwargs):
    ext_modules = [
        Pybind11Extension("HEAAN",
            ["bind/base64.cpp", "bind/heaan_wrapper.cpp"],
            # Example: passing in the version to the compiled code
            include_dirs=[py_include_dir,
            'pybind11/include',
            '/usr/local/include',
            'HEAAN/src'],
            language='c++',
            extra_compile_args=['-std=c++17'],
            extra_objects=['/usr/local/lib/libntl.so', 'HEAAN/lib/libHEAAN.a'],
            #define_macros = [('VERSION_INFO', __version__)],
            package_dir = {'': 'hemul/'},
            )]
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmd_class": {"build_ext": build_ext},
        "zip_safe": False,
    })