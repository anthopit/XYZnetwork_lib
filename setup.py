from setuptools import setup, find_packages


setup(
    name='xyznetwork_lib',
    version='0.0.2',
    license='MIT',
    author="Anthony Pitra",
    author_email='anthony.pitra01@gmail.com',
    packages=find_packages('xyznetwork'),
    package_dir={'': 'xyznetwork'},
    url='https://github.com/anthopit/XYZnetwork_lib',
    keywords='network graph visualization transportation railway GNN',
    install_requires=[
          'setuptools',
          'pandas',
          'numpy',
          'matplotlib',
          'node2vec',
          'torch',
          'networkx',
          'tqdm',
          'geopy',
          'seaborn',
          'scipy',
          'scikit-learn',
          'plotly',
          'distinctipy',
          'torch_geometric'
      ],

)