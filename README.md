# Chaotic Systems Prediction: Deep Learning vs Traditional Methods

Research comparing deep learning and traditional methods for predicting chaotic dynamical systems.

## Repository Structure

```
chaotic-systems-prediction/
├── README.md                    # This file
├── environment.yml              # Conda environment dependencies
├── requirements.txt             # Pip dependencies
├── setup.py                     # Package installation
├── .gitignore                   # Git ignore rules
│
├── config/                      # Configuration files
│   ├── models/                  # Model configurations (LSTM, Transformer, etc.)
│   ├── systems/                 # Chaotic system parameters (Lorenz, Rössler, etc.)
│   └── experiments/             # Full experiment configs
│
├── data/                        # Dataset storage
│   ├── raw/                     # Raw simulation/experimental data
│   ├── processed/               # Preprocessed datasets
│   ├── splits/                  # Train/val/test splits
│   └── README.md                # Data documentation
│
├── src/                         # Source code
│   ├── systems/                 # Chaotic system implementations
│   ├── traditional/             # Traditional methods (Reservoir Computing, Koopman, etc.)
│   ├── deep_learning/           # DL architectures (LSTM, Transformer, Neural ODE, etc.)
│   ├── utils/                   # Metrics, visualization, data utilities
│   └── training/                # Training and evaluation code
│
├── scripts/                     # Executable scripts
│   ├── generate_data.py         # Generate chaotic trajectories
│   ├── train.py                 # Train models
│   ├── evaluate.py              # Evaluate models
│   └── reproduce_paper.py       # Reproduce results
│
├── jobs/                        # HPC job scripts
│   ├── frontera/                # SLURM scripts for Frontera
│   └── vista/                   # SLURM scripts for Vista
│
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_visualization.ipynb
│   └── 03_results_summary.ipynb
│
├── experiments/                 # Individual experiment outputs
│   └── [experiment_name]/       # One folder per experiment
│       ├── config.yaml          # Experiment configuration
│       ├── checkpoints/         # Model checkpoints
│       ├── logs/                # Training logs
│       └── results/             # Predictions and metrics
│
├── results/                     # Aggregated results
│   ├── figures/                 # Publication figures
│   ├── tables/                  # Performance tables
│   └── benchmarks.csv           # Summary of all methods
│
└── tests/                       # Unit tests
    ├── test_systems.py          # Test chaotic systems
    └── test_models.py           # Test model implementations
```

## Directory Descriptions

- **`config/`** - YAML/JSON configuration files for reproducible experiments
- **`data/`** - All datasets (consider symlinking to HPC scratch space for large files)
- **`src/`** - Core implementations of systems, models, and training logic
- **`scripts/`** - Command-line scripts to run experiments
- **`jobs/`** - SLURM batch scripts for TACC (Frontera/Vista)
- **`notebooks/`** - Interactive analysis and visualization
- **`experiments/`** - Output directory for individual experiment runs
- **`results/`** - Aggregated results across all experiments
- **`tests/`** - Unit tests for validation
