# Initialize conda for bash
eval "$(conda shell.bash hook)"

conda create -n pyml python=3.9 -y
conda activate pyml

conda install numpy=1.21.2 scipy=1.7.0 scikit-learn=1.0 matplotlib=3.4.3 pandas=1.3.2 -y --channel conda-forge