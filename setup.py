from setuptools import setup, find_packages

setup(
    name='vizhelper',
    version='0.1.0',
    description='Helpers for data visualization and classification metric plotting',
    author="ZabakJL",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ZabakJL/vizhelper',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
