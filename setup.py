from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="sgt_igraph",
            sources=["modules/sgt_base.c", "modules/sgt_module.c"],
            libraries=["igraph"],
            library_dirs=["modules/igraph_lib/lib"],
            # extra_objects=["modules/igraph_lib/lib/libigraph.a"],
            include_dirs=["modules/igraph_lib/include/igraph"],
            extra_compile_args=["-fdeclspec"]  # Add the compiler flag here
        ),
    ]
)
