# Differentiable Simulation Tutorials

[![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![JAX](https://img.shields.io/badge/JAX-9B59B6)](https://github.com/jax-ml/jax) [![JAX-FEM](https://img.shields.io/badge/JAX--FEM-0.0.12-1f77b4)](https://github.com/deepmodeling/jax-fem) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

Companion code for the review paper *A Review of Differentiable Programming in Computational Mechanics*. The two notebooks illustrate the two common ways of differentiable programming: explicit differentiation through an explicit time-stepping recurrence, and implicit differentiation of an implicitly defined equilibrium. Both implementations are written in [JAX](https://github.com/jax-ml/jax), with the FEA support from [JAX-FEM](https://github.com/deepmodeling/jax-fem).



## Repository layout

```text
tutorial_diffsim/
├── README.md                     
├── assets/        
├── data/                         # input and output files      
├── requirements.txt              # additional tutorial dependencies
├── tutorial_explicit.ipynb       # Tutorial 1: explicit differentiation + checkpointing
├── tutorial_implicit.ipynb       # Tutorial 2: implicit differentiation of neural elastoplasticity            
```

## Installation

First follow the [JAX-FEM installation instructions](https://deepmodeling.github.io/jax-fem/guide/Installation.html), including its [JAX](https://github.com/google/jax) setup. Then clone this repository and install the additional tutorial dependencies in the same environment:

```bash
git clone https://github.com/xwpken/tutorial_diffsim.git
cd tutorial_diffsim

conda activate jax-fem-env  # replace with your JAX-FEM environment name

pip install -r requirements.txt
jupyter lab tutorial_explicit.ipynb
```

## Tutorial examples

| | Tutorial 1 | Tutorial 2 |
| :--- | :--- | :--- |
| Notebook | [`tutorial_explicit.ipynb`](tutorial_explicit.ipynb) | [`tutorial_implicit.ipynb`](tutorial_implicit.ipynb) |
| Problem | Free vibration of a 3D elastic waveguide | Quasi-static elastoplasticity of a dog-bone specimen |
| Forward solver | Explicit central difference with row-sum mass lumping | Newton–Raphson with a return-mapping constitutive update |
| Discretization | 1,280 `HEX8` elements (2,025 nodes, 6,075 DOF) | 7,900 `TET4` elements (1,706 nodes, 5,118 DOF) |
| Target | Recover the spatial Young's modulus field | Identify the neural yield function |
| Unknowns | 80 layerwise Young's moduli | 4,417 neural-network weights and biases |
| Gradient | Explicit differentiation through the time-stepping recurrence | Implicit differentiation of the equilibrium equations |

### Tutorial 1 — Explicit differentiation of transient elastodynamics

<p align="center">
  <img src="assets/example_explicit.png" width="80%" alt="Three-dimensional elastic waveguide under longitudinal free vibration. (a) Model setup. (b) Extended and (c) contracted configurations.">
  <br>
  <sub>Three-dimensional elastic waveguide under longitudinal free vibration. (a) Model setup. (b) Extended and (c) contracted configurations.</sub>
</p>

[`tutorial_explicit.ipynb`](tutorial_explicit.ipynb) identifies the axial distribution of Young's modulus in a three-dimensional elastic waveguide from transient displacement observations. Reverse-mode AD differentiates through explicit time integration, and a checkpointing comparison illustrates the memory–runtime trade-off.

<p align="center">
  <img src="assets/explicit_inverse.png" width="100%" alt="Post-processing of the identification in two panels">
  <br>
  <sub>Free-end displacement histories and identification convergence.</sub>
</p>

<p align="center">
  <img src="assets/explicit_checkpoint.png" width="100%" alt="Peak process RSS and loss-and-gradient time for different checkpoint intervals">
  <br>
  <sub>Memory and runtime versus checkpoint interval.</sub>
</p>

### Tutorial 2 — Implicit differentiation of neural elastoplasticity

<p align="center">
  <img src="assets/example_implicit.png" width="80%" alt="Dog-bone specimen with displacement loading and the resulting synthetic surface-displacement observations.">
  <br>
  <sub>Dog-bone specimen with displacement loading and the resulting synthetic surface-displacement observations.</sub>
</p>

[`tutorial_implicit.ipynb`](tutorial_implicit.ipynb) identifies a neural yield function from displacement observations of a dog-bone specimen under tension, compression, and shear. Gradients are obtained by implicitly differentiating the equilibrium equations. This example adapts the setup of [Gavris and Sun (2025)](https://doi.org/10.1073/pnas.2508732122) to a finite element formulation.

<p align="center">
  <img src="assets/implicit_optimization.png" width="92%" alt="Target, initial and identified yield surfaces for the three parametric cases">
  <br>
  <sub>Target, initial and identified yield surfaces for the three cases.</sub>
</p>

## Citation

If you find this repository useful, please cite the review paper:

```bibtex
To be added
```
