# eprbase

`eprbase` is a low-level Python framework for developing custom simulation
routines for continuous-wave electron paramagnetic resonance (cwEPR).

The framework provides CPU and GPU implementations of its modules with
compatible interfaces. This allows individual parts of a simulation to be
executed on either the CPU or GPU and combined as needed.

The code is fully object-oriented. Classes can be extended or modified through
inheritance, making it possible to implement custom functionality for
individual simulation requirements.

Currently, the Hamiltonian implementation is limited to radical pairs.
Support for arbitrary spin systems will be added in future releases.

## Features

- Low-level building blocks for custom cwEPR simulation routines
- CPU and GPU implementations with compatible interfaces
- Flexible hybrid CPU/GPU simulation workflows
- Object-oriented design
- Extensible and modifiable classes through inheritance
- Numerical calculations based on NumPy and SciPy
- GPU acceleration using CuPy
- Current support for radical-pair Hamiltonians
- Planned support for arbitrary spin systems

## Requirements

- Python 3.13 or newer

## Installation

### Using uv

#### CPU version

```bash
uv add eprbase
```

#### GPU version

```bash
uv add eprbase --extra gpu
```

The GPU version requires a compatible NVIDIA GPU and CUDA installation.

### Using pip

#### CPU version

```bash
pip install eprbase
```

#### GPU version

```bash
pip install eprbase[gpu]
```

The GPU version requires a compatible NVIDIA GPU and CUDA installation.

## Development

Clone the repository and install the development dependencies with `uv`:

```bash
git clone https://github.com/florianquintes/eprbase.git
cd eprbase
uv sync --dev
```

Install the optional GPU dependencies:

```bash
uv sync --extra gpu
```

Run the tests:

```bash
uv run pytest
```

Check the code formatting and linting:

```bash
uv run ruff format --check .
uv run ruff check .
```

Format the code locally:

```bash
uv run ruff format .
```

## Documentation

The documentation is available on
[GitHub Pages](https://florianquintes.github.io/eprbase/).

## License

This project is licensed under the GNU General Public License v3.0.
See the [LICENSE](LICENSE) file for details.
