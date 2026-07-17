Introduction
============

Overview
--------

``eprbase`` is a low-level Python framework for developing custom simulation
routines for continuous-wave electron paramagnetic resonance (cwEPR).

It provides object-oriented building blocks for constructing and extending
simulation workflows. The classes can be customized through inheritance, which
allows users to adapt existing functionality or implement specialized
simulation methods.

CPU and GPU implementations
----------------------------

All modules are available in both CPU and GPU variants. Their
interfaces are designed to be compatible, allowing different parts of a
simulation to run on the CPU or GPU as required.

The CPU implementation uses NumPy and SciPy. GPU acceleration is provided
through the optional CuPy dependency.

Module overview
---------------

The following table provides an overview of the main modules:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Description
   * - ``grid``
     - Provides classes and functionality for creating and handling simulation
       grids.
   * - ``hamiltonian``
     - Provides classes for constructing and handling Hamiltonian operators.
       The current implementation is limited to radical-pair systems.
   * - ``interpolation``
     - Provides functions for interpolating between magnetic field points or
       orientations.
   * - ``resonance_fields``
     - Provides functionality for calculating resonance fields.
   * - ``spectra``
     - Provides functionality for constructing spectra from the calculated
       resonance fields.

GPU modules
-----------

The GPU implementations are located in the ``eprbase.gpu`` package. They
provide GPU-enabled counterparts to the CPU modules:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - CPU module
     - GPU module
   * - ``eprbase.grid``
     - ``eprbase.gpu.grid``
   * - ``eprbase.hamiltonian``
     - ``eprbase.gpu.hamiltonian``
   * - ``eprbase.interpolation``
     - ``eprbase.gpu.interpolation``
   * - ``eprbase.resonance_fields``
     - ``eprbase.gpu.resonance_fields``
   * - ``eprbase.spectra``
     - ``eprbase.gpu.spectra``

Extensibility
-------------

The framework is designed to be extended through object-oriented programming.
Existing classes can be subclassed to add new functionality or change selected
parts of an algorithm.