from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="sgt",
            sources=["modules/sgtmodule.c", "modules/sgt_base.c"],
            libraries=["igraph"],
            # library_dirs=["modules/igraph_lib/lib"],
            library_dirs=["/opt/homebrew/Cellar/igraph/0.10.10/lib"],
            # extra_objects=["modules/igraph_lib/lib/libigraph.a"],
            # include_dirs=["modules/igraph_lib/include/igraph"],
            include_dirs=["/opt/homebrew/Cellar/igraph/0.10.10/include/igraph"],
            # extra_compile_args=["-fdeclspec"]  # Add the compiler flag here
        ),
    ]
)
