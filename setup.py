from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="sgt_igraph",
            sources=["modules/sgt_base.c", "modules/sgt_modules.c"],
            libraries=["igraph"],
            library_dirs=["modules/igraph_lib/lib"],
            include_dirs=["modules/igraph_lib/include/igraph"]
        ),
    ]
)
