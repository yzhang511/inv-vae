<a name="readme-top" id="readme-top"></a>

<!-- PROJECT LOGO -->

<div width="100" align="right">
<a href="https://github.com/yzhang511/inv-vae">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/yzhang511/inv-vae/blob/main/assets/inv_vae_icon.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/yzhang511/inv-vae/blob/main/assets/inv_vae_icon.png">
      <img alt="Logo toggles light and dark mode" src="https://github.com/yzhang511/inv-vae/blob/main/assets/inv_vae_icon.png"  width="100" align="right">
    </picture>
</a>
</div>


## Invariant variational auto-encoding

**[Motivation]** Mapping of human brain structural connectomes via diffusion MRI offers a unique opportunity to understand brain structural connectivity and relate it to various human traits, such as cognition. However, the presence of motion artifacts during image acquisition can compromise the accuracy of connectome reconstructions and subsequent inference results. 

**[Method]** We develop a generative model to learn low-dimensional representations of structural connectomes that are invariant to motion artifacts, so that we can link brain networks and human traits more accurately, and generate motion-adjusted connectomes:

<div align="right">
<a href="https://github.com/yzhang511/inv-vae">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/yzhang511/inv-vae/blob/main/assets/model_diagram.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/yzhang511/inv-vae/blob/main/assets/model_diagram.png">
      <img alt="Logo toggles light and dark mode" src="https://github.com/yzhang511/inv-vae/blob/main/assets/model_diagram.png"  align="right">
    </picture>
</a>
</div>

**[Full paper]** [Motion-Invariant Variational Auto-Encoding of Brain Structural Connectomes](https://arxiv.org/abs/2212.04535).

**[Note]** This model is not limited to removing motion from the neuroimaging data. You can adapt the code to remove the impact of arbitrary nuisance variables on your data.

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## ⏳ Installation
Create an environment and install this package and any other dependencies:
```
conda create -n inv_vae python=3.8
conda activate inv_vae
git clone https://github.com/yzhang511/inv_vae.git
cd inv_vae
pip install -e.
```
<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## ⚡️ Quick Start
Example usage can be found in [demo](https://github.com/yzhang511/inv-vae/tree/main/demo).

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

<!-- LICENSE -->
## :classical_building: License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">Back to top</a>)</p>

## ✏️ Cite Us

If you found the paper useful, please cite us:
```
@article{zhang2022motion,
  title={Motion-Invariant Variational Auto-Encoding of Brain Structural Connectomes},
  author={Zhang, Yizi and Liu, Meimei and Zhang, Zhengwu and Dunson, David},
  journal={arXiv preprint arXiv:2212.04535},
  year={2022}
}
```
<p align="right">(<a href="#readme-top">Back to top</a>)</p>




