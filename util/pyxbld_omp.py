import numpy as np
import platform
import os.path

extra_link_args = ['-fopenmp']
# work around problems on OSX with bad linkers
if platform.system() == 'Darwin':
    if os.path.exists('/usr/local/lib/libiomp5.dylib'):
        extra_link_args = ['-L/usr/local/lib', '-liomp5']

def make_ext(modname, pyxfilename):
    from distutils.extension import Extension
    return Extension(name=modname,
                     sources=[pyxfilename],
                     extra_compile_args=['-Wno-unused-function',
                                         '-std=gnu99',
                                         '-Ofast',
                                         '-march=native',
                                         '-fopenmp',
                                         '-I{}'.format(np.get_include()),
                                         '-I.'],
                     extra_link_args=extra_link_args)
