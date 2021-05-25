# Introduction to experiments

## In this folder the experiments for the Polar model on the MNIST and the MNIST-rot datasets are available

### Polar Model
- "Polar_MNIST.ipynb" is a jupyter notebook for training on non-rotated MNIST, tested on
   randomly rotated MNIST
- "Polar_MNIST-r.ipynb" carries out the experiment for training and testing on rotated MNIST, denoted as MINST-r in the paper.
- "Polar_ROT_MNIST.ipynb" contains the polar model trained and evaluated on the MNIST-ROT datasets
- "Polar_CelebA.ipynb" contains the polar model trained on non-rotated CelebA, tested on randomly rotated CelebA
- "Polar_CelebA-r.ipynb" contains the polar model trained and tested on randomly rotated CelebA
- "Polar_Visualisations.ipynb" contains visualisations of the images before and in polar

### G-Conv Model
- The G-Conv model can be found here from the original github: https://github.com/tscohen/GrouPy
- And its original experiments are here: https://github.com/tscohen/gconv_experiments
- We used an endorsed by the author version in pytorch here: https://github.com/adambielski/pytorch-gconv-experiments

### Celeb Experiments
- The dataset is available here: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Structure of experiments is similar to above notebooks
- Just only consider a binary classification on attribute: "Attractive" as mentioned in paper

### Requirements
- We used Python 3.8.5
- We have attatched a requirements.txt file for convenience

