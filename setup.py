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
          'setuptools~=67.4.0',
          'pandas~=2.0.1',
          'numpy~=1.24.3',
          'matplotlib~=3.7.1',
          'node2vec~=0.4.6',
          'torch~=2.0.0',
          'networkx~=2.8.8',
          'tqdm~=4.65.0',
          'geopy~=2.3.0',
          'seaborn~=0.12.2',
          'scipy~=1.10.1',
          'scikit-learn~=1.2.2',
          'plotly~=5.14.1',
          'distinctipy~=1.2.2',
          'torch_geometric'
      ],

)