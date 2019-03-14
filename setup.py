import os
from distutils.core import setup

install_requires=[
    "numpy",
    "nltk",
    "textblob",
    "keras",
    "pandas",
]

setup_requires=[
    "numpy",
]

extras_require = {
    "fasttext": ["fasttext"],
}

setup(
    name="ontokom",
    version="0.1",
    description="",
    url="",
    author="Robin Kahlow",
    license="MIT",
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    packages=["ontokom"],
)
