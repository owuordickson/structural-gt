from setuptools import Extension, setup

setup(
    # package_data={"StructuralGTb": ["c_modules/lib_pthreads/bin/*w64.dll"]},
    ext_modules=[
        Extension(
            name="sgt",
            # sources=["src/StructuralGTb/c_modules/testmodule.c"],
            sources=["src/StructuralGTb/c_modules/sgtmodule.c", "src/StructuralGTb/c_modules/sgt_base.c"],
            # libraries=["igraph"],  # macOS and Linux
            # include_dirs=["/opt/homebrew/Cellar/igraph/0.10.10/include/igraph"],  # macOS and Linux
            # library_dirs=["/opt/homebrew/Cellar/igraph/0.10.10/lib"],  # macOS and Linux
            libraries=["igraph-x64", "libpthreadVC3-x64"],  # Windows
            include_dirs=["C:/MinGW/include/igraph", "C:/MinGW/include/pthread"],  # Windows
            library_dirs=["C:/MinGW/lib/igraph", "C:/MinGW/lib/pthread"],  # Windows
            extra_link_args=["/VERBOSE:LIB"],
        )
    ]
    # data_files=[("", ["C:/MinGW/bin/pthread/pthreadVC3-w64.dll"])]
)
