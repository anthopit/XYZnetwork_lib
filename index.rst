.. XYZnetwork documentation master file, created by
   sphinx-quickstart on Wed Apr 19 17:26:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XYZnetwork's documentation!
######################################

.. toctree::
   :maxdepth: 2
   :caption: Contents:

What is XYZnetwork?
-------------------

XYZnetwork is the name of our team. It is also the name of the library
you have just installed! This library will help you create, manage and
analyse transport networks. The main goal of this library is for
researching ways of optimising and improving current public transport
methods and networks.

In this document, we will go over the different modules contained in the
library and explain their purpose and functionalities.

Modules
-------

There are 5 modules in this library:

* Preprocessing
* :ref:`char-ref`
* :ref:`clus-ref`

   * :ref:`embed-ref`
   * :ref:`MLclus-ref`

* :ref:`visu-ref`
* :ref:`deepl-ref`

   * GNN framework
   * Self-supervised training
   * Transfer learning
   * Advanced clustering

.. _visu-ref:

Visualisation
"""""""""""""

..
   Visualisation methods and parameters:

   -----

   .. py:function:: convert_minutes_to_ddhhmm(minutes)

      A utility function, converts minutes to a date format (day, hour, minute).

   Parameters:
      minutes(int): Number of minutes

   Returns:
      string: a string with the formatted date.

   Example:::

      visualisation.visualisation.convert_minutes_to_ddhhmm(153)
      > '00:02:33'

   Reference:
      | `Link to reference <https://stackoverflow.com/>`_
      | Link to reference `reflink`_

   .. _reflink: https://stackoverflow.com/

   -----

   .. py:function:: get_gradient_color(value)

      Returns a color from a gradient based on a given value.

   Parameters:
      | value (float): The input value to use for determining the color.
      | cmap_name (str): The name of the Matplotlib colormap to use.

   Returns:
      tuple: A tuple representing the RGB values of the color at the given value on the gradient.

-----

.. automodule:: visualisation.visualisation
   :members:
