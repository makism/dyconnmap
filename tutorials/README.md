# Requirements

Check `dyconnmap`'s requirements. Additionally you will need `jupyter-lab`.

Install the requirements with `pip install -r requirements.txt`.

# How to run (sample instructions)

From a terminal, change to the `tutorials/` subdirectory.

If you haven't installed `dyconnmap` using pip, you need to:
```
export PYTHONPATH=$PWD/../
```
Then, start a Jupyter Lab environment in the local directory with:
```
JUPYTER_CONFIG_DIR=$PWD jupyter-lab
```

and you are set to go over the individual notebooks. It is recommended that you start from the very first ones, because they produce derivatives that used in later notebooks.

# fMRI data
The fMRI time series are taken from the public dataset (around 1000 participants) provided from the "IMPAC - IMaging-PsychiAtry Challenge: predicting autism - A data challenge on Autism Spectrum Disorder detection" (https://paris-saclay-cds.github.io/autism_challenge/).
To acquire the data, you will have to follow the instructions found in the starting kit (http://nbviewer.jupyter.org/github/ramp-kits/autism/blob/master/autism_starting_kit.ipynb). We are using the fMRI time series fitted for the MSDL atlas, so retrieve the relevant data.

# EEG data
The EEG data were taken from https://physionet.org/content/eegmmidb [1, 2].
You can fetch them easily from the notebook.

# References

1. Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004. [In 2008, this paper received the Best Paper Award from IEEE TBME.]

2. Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/cgi/content/full/101/23/e215]; 2000 (June 13).
