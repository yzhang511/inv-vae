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

**[Motivation]** Mapping of human brain structural connectomes via diffusion MRI offers a unique opportunity to understand brain structural connectivity and relate it to various human traits, such as cognition. However, the presence of motion artifacts during image acquisition can compromise the accuracy of connectome reconstructions and subsequent inference results. We develop a generative model to learn low-dimensional representations of structural connectomes that are invariant to motion artifacts, so that we can link brain networks and human traits more accurately, and generate motion-adjusted connectomes:

<div align="right">
<a href="https://github.com/yzhang511/inv-vae">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/yzhang511/inv-vae/blob/main/assets/model_diagram.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/yzhang511/inv-vae/blob/main/assets/model_diagram.png">
      <img alt="Logo toggles light and dark mode" src="https://github.com/yzhang511/inv-vae/blob/main/assets/model_diagram.png"  align="right">
    </picture>
</a>
</div>

**[Experiments]** We applied the proposed model to data from the Adolescent Brain Cognitive Development (ABCD) study and the Human Connectome Project (HCP) to investigate how our motion-invariant connectomes facilitate understanding of the brain network and its relationship with cognition. Empirical results demonstrate that the proposed motion-invariant variational auto-encoder (inv-VAE) outperforms its competitors in various aspects. In particular, motion-adjusted structural connectomes are more strongly associated with a wide array of cognition-related traits than other approaches without motion adjustment.

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




