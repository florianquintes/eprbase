.. eprbase documentation master file, created by
   sphinx-quickstart on Thu Jul 16 14:52:19 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

eprbase
=======

``eprbase`` is a low-level Python framework for developing custom simulation
routines for continuous-wave electron paramagnetic resonance (cwEPR).

The framework provides compatible CPU and GPU implementations. This allows
individual parts of a simulation to be executed on either the CPU or GPU and
combined as required. Its object-oriented design also enables existing classes
to be extended or customized through inheritance.

.. important::

   The Hamiltonian implementation currently supports radical-pair systems
   only. Support for arbitrary spin systems is planned for future releases.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   introduction
   installation
   examples
   api
   development
   contributing
   license