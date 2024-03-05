from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="sgt",
            sources=["c_modules/testmodule.c"],
            # sources=["c_modules/sgtmodule.c", "c_modules/sgt_base.c"],
            # libraries=["igraph"],  # macOS and Linux
            # include_dirs=["/opt/homebrew/Cellar/igraph/0.10.10/include/igraph"],  # macOS and Linux
            # library_dirs=["/opt/homebrew/Cellar/igraph/0.10.10/Libs"],  # macOS and Linux
            libraries=["igraph"],  # Windows
            include_dirs=["C:/MinGW/include/igraph", "C:/MinGW/include/pthread"],  # Windows
            library_dirs=["C:/MinGW/lib/igraph", "C:/MinGW/lib/pthread", "C:/MinGW/bin/pthread"],  # Windows
            # extra_compile_args=["-std=c99", "-fdeclspec"]  # Add the compiler flag here
        ),
    ]
)
