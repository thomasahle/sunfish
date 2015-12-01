import platform
import os.path

def install():
    if platform.system() == 'Darwin':
        # most OSX machines now use clang-based compilers.
        # clang only recently got support for OpenMP.
        # search for a good compiler
        if os.path.exists('/usr/local/bin/gcc'):
            os.environ['CC'] = '/usr/local/bin/gcc'
            print("Compiling with /usr/local/bin/gcc")
        elif os.path.exists('/usr/local/bin/clang-omp'):
            os.environ['CC'] = '/usr/local/bin/clang-omp'
            print("Compiling with /usr/local/bin/clang-omp")
        else:
            print("No definitely good compiler found, things may not work.")
