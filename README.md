# motion-invariant variational auto-encoding of brain structural connectomes

create conda environment and install packages:
```
conda create -n inv_vae python=3.8
conda install --file requirements.txt
```

for jupyter lab (need to install `nodejs`):
```
pip install jupyterlab jupytext
conda install -c "conda-forge/label/cf202003" nodejs
conda update nodejs
jupyter lab build
```

then serve:
```
mkdir -p .jupter/lab/workspaces
JUPYTERLAB_WORKSPACES_DIR=.jupyter/lab/workspaces jupyter lab --no-browser --ip=0.0.0.0
```

----

demo:
- contains instructions on how to use inv-vae and gate

to do:
- add R code used to visualize brain connections using a circular graph
- add R code used for trait prediction using SOG
- add python code used for trait prediction using ComBat
- add notebooks to reproduce results in the paper

