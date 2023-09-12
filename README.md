# SynthNN
This repository is the official implementation of SynthNN that is described in the paper "Predicting the Synthesizability of Crystalline Inorganic Materials from the Data of Known Material Compositions". (https://www.nature.com/articles/s41524-023-01114-4)
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
  - [Retrain SynthNN](#retrain-synthnn)
- [Data](#data)
- [Authors](#authors)
- [License](#license)

## How to cite
If you use SynthNN, please cite the following work:

```
@article{antoniuk_SynthNN_2023,
	title = {Predicting the synthesizability of crystalline inorganic materials from the data of known material compositions},
	volume = {9},
	copyright = {2023 Springer Nature Limited},
	issn = {2057-3960},
	url = {https://www.nature.com/articles/s41524-023-01114-4},
	doi = {10.1038/s41524-023-01114-4},
	language = {en},
	number = {1},
	urldate = {2023-09-07},
	journal = {npj Computational Materials},
	author = {Antoniuk, Evan R. and Cheon, Gowoon and Wang, George and Bernstein, Daniel and Cai, William and Reed, Evan J.},
	month = aug,
	year = {2023},
	note = {Number: 1
Publisher: Nature Publishing Group},
	keywords = {Computational methods, Design, synthesis and processing},
	pages = {1--11},
}
```

## Prerequisites
Requirements:
- Python
- [Pymatgen](https://pymatgen.org/installation.html)
- [Tensorflow](https://www.tensorflow.org/install)

## Usage
### Reproduce Figures
All figures in the manuscript can be reproduced with the Figure_Reproduction Jupyter Notebook. Note that a significant number of the figures depend on having access to the full ICSD dataset, which cannot be shared in this repo due to ICSD License Agreement. However, all figures can be reproduced with the provided pre-processed data in the 'Figure_data' folder.

### Predict Synthesizability
Predicting the synthesizability of a material composition with a pre-trained version of SynthNN can be done with SynthNN_predict.ipynb.
We recommend referring to the below performance metrics when choosing a decision threshold to label a material as synthesizable or not. The below table indicates the performance of
SynthNN of a dataset with a 20:1 ratio of unsynthesized:synthesized examples. Note, a threshold value of '0.10' means that any material with a SynthNN output greater than 0.10 is taken to be synthesizable, which leads to low precision but high recall.
Threshold | Precision | Recall | 
| :---: | :---: | :---: |
0.10 | 0.239 | 0.859 |
0.20 | 0.337 | 0.783 |
0.30 | 0.419 | 0.721 |
0.40 | 0.491 | 0.658 |
0.50 | 0.563 | 0.604 |
0.60 | 0.628 | 0.545 |
0.70 | 0.702 | 0.483 |
0.80 | 0.765 | 0.404 |
0.90 | 0.851 | 0.294 |

### Retrain SynthNN
A new SynthNN model can be trained from scratch with the train_SynthNN.ipynb Jupyter Notebook. Simply edit the 'positive_example_file_path' and 'negative_example_file_path' dictionary entries to point to your list of synthesized and unsynthesized materials, respectively. By default, trained model weights will be saved in the specified 'Trained_models' folder.

## Data
The Synthesizability Dataset used in this work was obtained from the [ICSD API](https://icsd.products.fiz-karlsruhe.de/en/products/icsd-products#icsd+api+service). If the ICSD API is not accessible, all figures can still be reproduced with the pre-processed data given in the "Figure_data" directory.

## Authors
This code was primarily written by Evan Antoniuk (antoniuk1@llnl.gov).

## License
SynthNN is released under the MIT License.
