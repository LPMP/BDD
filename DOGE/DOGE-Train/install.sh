# Need python>=3.10

conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning==2.0.2

# for GNN:
pip install torch_geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html

conda install -c conda-forge ecole # required to generate random problem instances.
pip install gurobipy # required for ground-truth generation.

pip install tqdm
pip install torchmetrics
pip install tensorboard

# For config files:
cd external/yacs
python setup.py install
cd ..

# Parallel deferred min-marginal averaging algorithm:
cd external/BDD
python setup.py install
cd ..