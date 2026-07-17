Installation
============

Requirements
------------

``eprbase`` requires Python 3.13 or newer.

Installation commands
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 
     - uv
     - pip
   * - CPU
     - ``uv add eprbase``
     - ``pip install eprbase``
   * - CPU + GPU
     - ``uv add eprbase[gpu]``
     - ``pip install eprbase[gpu]``

GPU requirements
----------------

GPU support requires:

* A compatible NVIDIA GPU
* A compatible CUDA environment
* The ``cupy-cuda13x`` package
* The ``pynvml`` package

The GPU modules are available through the ``eprbase.gpu`` package. ``cupy-cuda13x`` and ``pynvml`` are automatically installed when using the ``eprbase[gpu]`` installation option.