from distutils.core import setup, Extension

setup(
    name="ttmodule",
    version="0.2",
    description="""A lightweight Tensor Train extension for pytorch.nn""",
    license="MIT License",
    author="Ivan Nazarov",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=[
        "ttmodule",
    ],
    requires=["torch", "numpy"]
)
