Development
===========

Development setup
------------------

Clone the repository and install the development dependencies with ``uv``:

.. code-block:: bash

   git clone https://github.com/florianquintes/eprbase.git
   cd eprbase
   uv sync --dev

To install the optional GPU dependencies as well:

.. code-block:: bash

   uv sync --extra gpu

Running tests
-------------

Run the test suite with ``pytest``:

.. code-block:: bash

   uv run pytest

Code formatting and linting
---------------------------

Check the code formatting:

.. code-block:: bash

   uv run ruff format --check .

Check the code for linting issues:

.. code-block:: bash

   uv run ruff check .

Format the code automatically:

.. code-block:: bash

   uv run ruff format .

Build the documentation
-----------------------

Build the documentation locally with Sphinx:

.. code-block:: bash

   uv run sphinx-build -b html src/eprbase/docs/source \
      src/eprbase/docs/build/html