.. raw:: html

    <h1>Welcome to the XYZnetwork documentation!</h1>

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    DEMO

*******************
What is XYZnetwork?
*******************

XYZnetwork is the name of our team. It is also the name of the library
you have just installed! This library will help you create, manage and
analyse transport networks. The main goal of this library is for
researching ways of optimising and improving current public transport
methods and networks.

In this document, we will go over the different modules contained in the
library and explain their purpose and functionalities.

In the Contents table, you will find links to a demonstration jupyter notebook.

****************************
What is a transport network?
****************************

A transport network is a network of a public transport. For instance, we
will be using the Chinese railway dataset. This dataset contains data
relative to the railway network of China. That is, trains, stations and
the different routes linking them, along with departure and arrival times,
distances and duration.

These networks will be represented by our main class, TransportNetwork:

.. automodule:: classes.transportnetwork
    :members:
    :undoc-members:

This class will be used for plotting, mapping and analysing its graph, which
represents the dataset's network. This will be done via our modules.

*******
Modules
*******

There are 5 modules in this library:

* :ref:`prep-ref`
* :ref:`char-ref`
* :ref:`ML-ref`

   * :ref:`embed-ref`
   * :ref:`clus-ref`

* :ref:`visu-ref`
* :ref:`deepl-ref`

   * :ref:`DLmodel-ref`
   * :ref:`DLdata-ref`
   * :ref:`DLloss-ref`
   * :ref:`DLrun-ref`


.. _prep-ref:

Preprocessing
-------------

.. automodule:: preprocessing.Preprocessing
    :members:
    :undoc-members:

.. _char-ref:

Characterisation
----------------

.. automodule:: characterisation.assortativity
    :members:
    :undoc-members:

.. automodule:: characterisation.centrality
    :members:
    :undoc-members:

.. automodule:: characterisation.clustering
    :members:
    :undoc-members:

.. automodule:: characterisation.degree
    :members:
    :undoc-members:

.. automodule:: characterisation.distance
    :members:
    :undoc-members:

.. automodule:: characterisation.page_rank
    :members:
    :undoc-members:

.. automodule:: characterisation.path
    :members:
    :undoc-members:

.. _ML-ref:

Machine Learning
----------------

.. _embed-ref:

Embedding
"""""""""

.. automodule:: ML.embedding
    :members:
    :undoc-members:

.. _clus-ref:

Clustering
""""""""""

.. automodule:: clustering.cluster
    :members:
    :undoc-members:

.. _visu-ref:

Visualisation
-------------

.. automodule:: visualisation.visualisation
    :members:
    :undoc-members:

.. _deepl-ref:

Deep Learning
-------------

.. _DLmodel-ref:

Model
"""""

The model.py file contains classes of models to be used.

.. automodule:: GNN.model
    :members:
    :undoc-members:

.. _DLdata-ref:

Data
""""

.. automodule:: GNN.data
    :members:
    :undoc-members:

.. _DLloss-ref:

Loss
""""

.. automodule:: GNN.loss
    :members:
    :undoc-members:

.. _DLrun-ref:

Run
"""

.. automodule:: GNN.run
    :members:
    :undoc-members:
