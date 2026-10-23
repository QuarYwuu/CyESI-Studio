# CyESI Studio

CyESI Studio is an interactive Python application for exploring interfaces used in label-free mass cytometry coupling flow cytometry with electrospray ionization mass spectrometry (CyESI-MS). It connects capillary geometry, liquid and gas flow, solvent exchange, cell transport, electrical fields and mass-spectrometer acquisition in a local browser workspace.

The interface includes an animated 2D device overview, axisymmetric and 3D field views, individual cell trajectories, parameter studies and acquisition-window analysis. Numerical solvers and browser assets are included in the source.

## Installation

Use Python 3.10 or newer. Download or clone this repository, then open a terminal in its directory.

On Windows, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --studio
```

On Linux or macOS:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py --studio
```

Keep the terminal open while using the application. The interface opens in your browser at a local address. To stop the server, press `Ctrl+C` in the terminal.

An editable installation also provides the `cyesi` and `cyesi-studio` commands:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\cyesi-studio.exe
```

Numba acceleration is optional: install the package with `pip install -e ".[performance]"` using your virtual environment's Python.

## Using the application

1. In **Experiment**, set the capillary dimensions, recess, fluid supply, electrical settings and cell properties.
2. Run the selected solver. Use **Overview**, **Fields** and **Cell transport** to inspect cell motion, flow, pressure and methanol exposure.
3. In **MS acquisition**, adjust accumulation time, resolution and signal duration to compare single-source, mixed and missed collection opportunities.
4. Use **Parameter studies** to compare operating conditions and **Results** to reopen your own simulations or export data and figures.

With no existing results, the first launch computes a fresh 64-cell 2D preview using a reduced mesh and no transient spray calculation. Its settings remain editable; increase the mesh and response sample count for your application. The repository contains no archived simulation results.

Generated files are saved under `outputs/studio` in the launch directory. To use another location, set `CYESI_OUTPUT_DIR` before launching; Studio creates a `studio` subdirectory there. For example, in PowerShell:

```powershell
$env:CYESI_OUTPUT_DIR = "D:\CyESI_runs"
.\.venv\Scripts\python.exe main.py --studio
```

## Model scope

The application links numerical transport models to specified molecular or signal durations and acquisition windows. Predicted collection opportunities are distinct from experimentally measured cell detection or metabolite identification. Acquisition timing currently represents the original Q Exactive quadrupole-Orbitrap instrument; time-of-flight (TOF) performance has not been evaluated in this software.

## License

Released under the [MIT License](LICENSE).
