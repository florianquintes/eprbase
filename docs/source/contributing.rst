Contributing
============

Thank you for your interest in contributing to ``eprbase``. Contributions,
bug reports, and improvement suggestions are welcome.

Development setup
------------------

Clone the repository and install the development dependencies with ``uv``:

.. code-block:: bash

   git clone https://github.com/florianquintes/eprbase.git
   cd eprbase
   uv sync --dev

Install the optional GPU dependencies only if you are working on GPU
functionality:

.. code-block:: bash

   uv sync --extra gpu

Branches
--------

The ``main`` branch contains the stable version of the project. Development
work should be based on the ``develop`` branch.

Create a feature or bug-fix branch from ``develop``:

.. code-block:: bash

   git switch develop
   git pull
   git switch -c feature/short-description

Use descriptive branch names, for example:

* ``feature/add-new-interpolation-method``
* ``fix/grid-boundary-condition``
* ``docs/improve-installation-guide``

Pull requests
-------------

Before opening a pull request:

* Keep the pull request focused on one topic.
* Add or update tests for changed functionality.
* Update the documentation where necessary.
* Ensure that the code is formatted and passes the linter.
* Ensure that the complete test suite passes.
* Keep commits clear and descriptive.
* Do not commit generated files, virtual environments, or credentials.

Run the local checks before submitting a pull request:

.. code-block:: bash

   uv run pytest
   uv run ruff format --check .
   uv run ruff check .

Pull requests should target the ``develop`` branch unless they contain a
release-related change. The CI workflows must pass before a pull request can
be merged.

Code style
----------

Use ``ruff`` for formatting and linting. Format code locally with:

.. code-block:: bash

   uv run ruff format .

GPU contributions
-----------------

GPU-specific code should preserve interface compatibility with the
corresponding CPU implementation wherever possible.

GPU tests require a compatible NVIDIA GPU and CUDA environment. CPU tests must
remain runnable without the optional GPU dependencies.

Documentation
-------------

Documentation is written in English and built with Sphinx. Build it locally
with:

.. code-block:: bash

   uv run sphinx-build -b html src/eprbase/docs/source \
      src/eprbase/docs/build/html

When changing public APIs, update the relevant docstrings and API
documentation.