from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            name="sgt",
            # sources=["src/StructuralGTc/SGT-c/testmodule.c"],
            sources=["src/StructuralGTc/SGT-c/sgtmodule.c", "src/StructuralGTc/SGT-c/sgt_base.c"],
            libraries=["igraph"],  # macOS and Linux
            include_dirs=["/opt/homebrew/Cellar/igraph/0.10.11/include/igraph"],  # macOS and Linux
            library_dirs=["/opt/homebrew/Cellar/igraph/0.10.11/lib"],  # macOS and Linux
            # libraries=["igraph-x64", "libpthreadVC3-x64"],  # Windows
            # include_dirs=["C:/MinGW/include/igraph", "C:/MinGW/include/pthread"],  # Windows
            # library_dirs=["C:/MinGW/lib/igraph", "C:/MinGW/lib/pthread"],  # Windows
            # extra_link_args=["/VERBOSE:LIB"],  # Windows
            # IMPORTED LIBS
            # libraries=["igraph-x64", "libpthreadVC3-x64"],  # Windows
            # include_dirs=["src/StructuralGTc/SGT-c/libraries/include/igraph",
            #              "src/StructuralGTc/SGT-c/libraries/include/pthread"],
            # library_dirs=["src/StructuralGTc/SGT-c/libraries/static-libs/igraph",
            #              "src/StructuralGTc/SGT-c/libraries/static-libs/pthread"],
            # libraries=["igraph-x64"],  # MacOS
            # include_dirs=["src/StructuralGTc/SGT-c/libraries/include/igraph"],
            # library_dirs=["src/StructuralGTc/SGT-c/libraries/static-libs/igraph"],

        )
    ]
)
