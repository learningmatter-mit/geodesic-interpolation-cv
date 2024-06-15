# Geodesic interpolation for collective variables
[![arXiv](https://img.shields.io/badge/arXiv-2402.01542-84cc16)](https://arxiv.org/abs/2402.01542)
[![MIT](https://img.shields.io/badge/License-MIT-3b82f6.svg)](https://opensource.org/license/mit)

This repository contains the code and input files to reproduce the results of the paper "Learning Collective Variables with Synthetic Data Augmentation through Physics-inspired Geodesic Interpolation" by [Yang et al.](https://arxiv.org/abs/2402.01542)

## Installation
We tested the code with Python 3.10 and the packages in `requirements.txt`.
For example, you can create a conda environment and install the required packages as follows (assuming CUDA 11.8):
```bash
conda create -n geodesic-cv python=3.10
conda activate geodesic-cv
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

To deploy the learned CV in MD simulations, you need to build the PLUMED package with the pytorch and drr modules, and then build the GROMACS package with the PLUMED patch.
We tested our code with PLUMED 2.9.0 with libtorch 2.0.1 and GROMACS 2023.

## Steps to reproduce the results
All the commands assume that you are in the root directory of the repository.
Your GROMACS binary might be different from `gmx_mpi`, and you might have to adjust `mdrun` options according to your hardware.

### 1. Unbiased simulations
```bash
cd simulations/unbiased/unfolded
gmx_mpi mdrun -deffnm nvt -nsteps 25000000 -plumed plumed.dat
gmx_mpi trjconv -f nvt.xtc -pbc nojump -o trajout.xtc

cd ../folded
gmx_mpi mdrun -deffnm nvt -nsteps 25000000 -plumed plumed.dat
gmx_mpi trjconv -f nvt.xtc -pbc nojump -o trajout.xtc
```

### 2. Geodesic interpolation
```bash
python scripts/geodesic_interpolation.py \
    --xtc-unfolded simulations/unbiased/unfolded/trajout.xtc \
    --xtc-folded simulations/unbiased/folded/trajout.xtc \
    --num-interp 5000 \
    --output-dir simulations/interpolation
```

### 3. Train ML CV models
Please refer to the notebook `train_cv.ipynb` for training the ML CV models.

### 4. Enhanced sampling simulations
We provide an example for a single run using the TDA CV.
In our paper, we used all combinations of CVs and tpr files.
```bash
# Create a simulation directory
mkdir -p simulations/enhanced/TDA/nvt_0; cd simulations/enhanced/TDA/nvt_0

# Create symbolic links to the input files
ln -s ../../tpr_files/nvt_0.tpr nvt.tpr
ln -s ../../plumed_files/plumed_TDA.dat plumed.dat
ln -s ../../plumed_files/TDA.pt .

# Run the simulation
gmx_mpi mdrun -deffnm nvt -nsteps 500000000 -plumed plumed.dat
```

### 5. Delta F and PMF calculation
We also provide a single example for the case mentioned above.
We followed the metadynamics grid range and sigma values for each CV from Table S1 in the SI.
```bash
python scripts/compute_pmf.py \
    --colvar-file simulations/enhanced/TDA/nvt_0/COLVAR \
    --cv-thresh -8.5 8.5 \
    --sigma 0.20 \
    --save-path simulations/enhanced/TDA/nvt_0
```


## Citation
```
@misc{yang2024learning,
    title={Learning Collective Variables with Synthetic Data Augmentation through Physics-inspired Geodesic Interpolation},
    author={Soojung Yang and Juno Nam and Johannes C. B. Dietschreit and Rafael G{\'o}mez-Bombarelli},
    year={2024},
    eprint={2402.01542},
    archivePrefix={arXiv},
    primaryClass={physics.chem-ph}
}
```
