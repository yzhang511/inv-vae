# inv_vae
Motion-invariant variational auto-encoding of brain structural connectomes

Create conda environment and install packages:
```
conda create -n inv_vae python=3.8
conda install --file requirements.txt
```

For jupyter lab (install `node` if you have not done so):
```
pip install jupyterlab jupytext
conda install -c "conda-forge/label/cf202003" nodejs
conda update nodejs
jupyter lab build
```

Then serve:
```
mkdir -p .jupter/lab/workspaces
JUPYTERLAB_WORKSPACES_DIR=.jupyter/lab/workspaces jupyter lab --no-browser --ip=0.0.0.0
```
