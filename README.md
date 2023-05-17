# SynthNN
This repository is the official implementation of SynthNN that is described in the paper "Predicting the Synthesizability of Crystalline Inorganic Materials from the Data of Known Material Compositions".

This repository serves a few functions:

  i) Reproduce all the figures in the paper.
  
  ii) Obtain synthesizability predictions for a general composition for an inorganic crystalline material.
  
  iii) Train your own material synthesizability model.
 
 
## Table of Contents
- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Reproduce Figures](#reproduce-figures)
  - [Predict Synthesizability](#predict-synthesizability)
- [Data](#data)
- [Authors](#authors)
- [License](#license)

## How to cite
If you use SynthNN, please cite the following preprint:
https://doi.org/10.21203/rs.3.rs-2574875/v1

## Prerequisites
Requirements:
- Python
- [Pymatgen](https://pymatgen.org/installation.html)
- [Tensorflow](https://www.tensorflow.org/install)
Alternatively, a conda environment can be made with the provided environment.yml file.

## Usage
### Reproduce Figures
All figures in the manuscript can be reproduced with the Figure_Reproduction Jupyter Notebook. 

### Predict Synthesizability
Predicting the synthesizability of a material composition with a pre-trained version of SynthNN can be done either with SynthNN_predict.ipynb or by running SynthNN_predict.py.

## Data
The Synthesizability Dataset used in this work was obtained from the [ICSD API](https://icsd.products.fiz-karlsruhe.de/en/products/icsd-products#icsd+api+service). If the ICSD API is not accessible, all figures can still be reproduced with the pre-processed data given in the "Figure_data" directory. The negative examples are provided in the Datasets folder.

## Authors
This code was primarily written by Evan Antoniuk (antoniuk1@llnl.gov).

## License
SynthNN is released under the MIT License.
