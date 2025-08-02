# DroneDesignMoo
Multi-objective optimization for drone design.

<img width="482" height="462" alt="Picture1" src="https://github.com/user-attachments/assets/6d9f28e4-def5-4f64-8688-d312f48b1e41" />

---

## Features

- **Multi-objective optimization** using NSGA-II (via [pymoo](https://pymoo.org/))
- **Customizable drone design problem**: easily modify objectives, constraints, and variables
- **Interactive Jupyter notebook** for running and analyzing optimizations
- **Visualization tools**: 3D Pareto front, parallel coordinates, and variable plots
- **Modular codebase**: clean separation of problem definition, optimization, and postprocessing

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/DroneDesignMoo.git
cd DroneDesignMoo
```

### 2. Set up the environment

Create a conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate drone_moo_312
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Start Jupyter and open the main notebook:

```bash
jupyter notebook notebooks/drone_design_optimization.ipynb
```

---

## Project Structure

```
DroneDesignMoo/
├── DroneDesignMoo/           # Python package (problem, optimizer, postprocess modules)
├── notebooks/
│   └── drone_design_optimization.ipynb
├── environment.yml
├── setup.py
├── pyproject.toml
└── README.md
```

---

## Example Usage

In the notebook, you can:

- **Set optimization parameters** (generations, population size, etc.)
- **Run the optimizer**:
    ```python
    res = run_optimization(n_gen=15, pop_size=100, seed=42)
    ```
- **Visualize results**:
    ```python
    plot_pareto_front(res.F, feasible_indices, max_solutions)
    plot_decision_variables(X_best, variable_names, max_solutions)
    plot_parallel_coordinates(X, F, G, variable_names, max_solutions)
    ```

---

## Customization

- **Modify objectives or constraints**: Edit `DroneDesignMoo/problem.py`
- **Change optimization algorithm or parameters**: Edit `DroneDesignMoo/optimizer.py`
- **Add new plots or analyses**: Edit `DroneDesignMoo/postprocess.py` or extend the notebook

---

## Requirements

- Python 3.12 (or compatible)
- numpy, scipy, matplotlib, pandas, jupyter, pymoo

All dependencies are listed in `environment.yml` and `pyproject.toml`.

---

## License

MIT License

---

## Acknowledgments

- [pymoo](https://pymoo.org/) for evolutionary optimization algorithms
- Inspired by real-world drone design challenges

---

## Contributing

Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

---

**Happy optimizing!**