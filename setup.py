from setuptools import setup, Extension
import numpy as np

# Define extension module
audio_rl_module = Extension(
    'libaudio_rl',
    sources=['audio_engine_export.cpp'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-std=c++11', '-O3']  # Removed architecture-specific flags
)

# Setup package
setup(
    name="audio_rl",
    version="0.1",
    description="Audio RL Environment with Parallelization",
    ext_modules=[audio_rl_module],
    install_requires=[
        'numpy',
        'torch',
    ],
)