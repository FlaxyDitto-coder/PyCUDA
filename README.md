**PyCU**
=====

Minimal PyCU — Lightweight CPU/CUDA scaffold exposing a tiny tensor API.
This repository contains two distribution build variants: `pycu-cpu` (CPU
implementation) and `pycu-cuda` (CUDA detection + stubs). The import name for
Python code is `pycu` regardless of distribution name.

**Install**
-----------

Install from PyPI (if published):

 - **CPU package:** `pip install pycu-cpu`
 - **CUDA package:** `pip install pycu-cuda`

Install from the local wheels included in this repository:

```bash
# install CPU wheel (from repo)
pip install wheels/cpu/*.whl

# install CUDA wheel (from repo)
pip install wheels/cuda/*.whl
```

Build and install locally (development):

```bash
pip install --upgrade build
python -m build
pip install dist/*.whl
```

**Quick Example**
-----------------

```python
from pycu import tensor, cuda_available, get_cuda_version

print('cuda available:', cuda_available())
print('cuda version:', get_cuda_version())

# CPU tensors
a = tensor([[1,2],[3,4]], device='cpu')
b = tensor([[5,6],[7,8]], device='cpu')
print('add:', (a + b).tolist())
print('mul:', (a * b).tolist())
print('matmul:', a.matmul(b).tolist())
```

If you request a CUDA tensor and CUDA isn't available, the library raises an
error. The current `CUDATensor` included in `pycu` is a stub placeholder; real
GPU kernels require native extensions.

**API / Commands**
------------------

- `tensor(data, device='cpu')` : Factory. Create a `Tensor` on `cpu` or `cuda`.
	- `data` is a nested Python list (e.g. `[[1,2],[3,4]]`).
	- `device` accepts `'cpu'` or `'cuda'`.

- `cuda_available()` -> bool : Returns True if the CUDA driver appears present.

- `get_cuda_version()` -> int|None : Returns the driver version number (if
	available) or `None`.

- `Tensor` (CPU implementation) methods:
	- `tolist()` : Get nested lists back.
	- `__add__(other)` : elementwise addition (other may be scalar or Tensor).
	- `__mul__(other)` : elementwise multiplication.
	- `matmul(other)` : 2D matrix multiplication (raises on wrong shapes).

**Examples and patterns**
-------------------------

Elementwise operations support broadcasting only for scalars; for two tensors
they must have identical shapes:

```python
x = tensor([[1,2],[3,4]])
y = tensor([[5,6],[7,8]])
z = x + y           # elementwise
s = x * 2           # scalar multiply
```

Matrix multiply example:

```python
x = tensor([[1,2],[3,4]])
y = tensor([[5,6],[7,8]])
r = x.matmul(y)     # [[19,22],[43,50]]
```

CUDA notes
----------

- `cuda_available()` uses a ctypes probe of the system CUDA driver (`libcuda`)
	and attempts minimal driver initialization where available.
- The included `CUDATensor` is a placeholder and raises `NotImplementedError`.
	Implementing real CUDA-backed tensors requires writing native extensions or
	using an existing GPU runtime.

**Wheels and GitHub release**
-----------------------------

This repository stores built artifacts under `wheels/cpu/` and `wheels/cuda/`.
To upload these artifacts to GitHub releases after pushing the repo, use:

```powershell
gh release create v0.1.0 --title "v0.1.0" --notes "Initial PyCU release" wheels/cpu/* wheels/cuda/*
```

**Troubleshooting**
-------------------

- If `pip` installation fails complaining about missing CUDA libraries when
	installing `pycu-cuda`, make sure the NVIDIA drivers are installed on the
	machine.
- If `twine` uploads to PyPI fail with 403, verify the API token scope and
	ownership for the package name.

**Contributing and Next Steps**
------------------------------

This project is a scaffold. If you want native GPU support we can:

- Add a backend using `cffi`/`ctypes` + CUDA kernels compiled to PTX, or
- Provide bindings with `pybind11`/C++ and ship wheels for specific platforms.

Pull requests, issues and feature requests welcome.


Publishing `pycu-cpu` and `pycu-cuda`
----------------------------------

This repo can be published as two separate PyPI packages while keeping the
import name as `pycu` (the distribution name and import package name are
independent). Two sample pyproject files are provided: `pyproject_cpu.toml`
and `pyproject_cuda.toml`.

Build and upload `pycu-cpu`:

```powershell
# switch to cpu pyproject
copy pyproject_cpu.toml pyproject.toml /Y
python -m build
# set env vars securely before upload
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "<YOUR_TOKEN>"
python -m twine upload --repository pypi --verbose dist/*
```

Build and upload `pycu-cuda`:

```powershell
# switch to cuda pyproject
copy pyproject_cuda.toml pyproject.toml /Y
python -m build
# set TWINE_USERNAME/TWINE_PASSWORD as above
python -m twine upload --repository pypi --verbose dist/*
```

Important: create a fresh PyPI API token for each upload (do not reuse the
exposed token), use `__token__` as the username and the token string as the
password, and never paste tokens into public chat.
