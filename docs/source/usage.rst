Usage
=====

.. _installation:

Installation
------------

To use pdc-dp-means, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pdc-dp-means


If you wish to install from source, you need to have `cython` installed, have `scikit-learn` installed on editable mode, and then:
1. Clone the repository.
2. softlink from the root dir to scikit-learn `sklearn` dir.
3. run `python setup.py build_ext --inplace`

