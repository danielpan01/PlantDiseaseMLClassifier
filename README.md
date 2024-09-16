# Plant Disease Identification using Neural Networks

![PlantDiseaseML](https://cdn.britannica.com/89/126689-004-D622CD2F/Potato-leaf-blight.jpg)

## Overview

This repository contains the code for a machine learning project focused on the automated identification of plant diseases using neural networks. The primary model used is a Multi-layer Perceptron (MLP) classifier, trained on the PlantVillage dataset. The project demonstrates the potential of neural networks in enhancing agricultural diagnostics by providing a scalable solution for early disease detection.

## Features

- **Data Preprocessing:** Resize, normalize, and prepare images for model training.
- **MLP Model Training:** Train a Multi-layer Perceptron classifier with hyperparameter tuning.
- **Model Evaluation:** Evaluate model performance using accuracy, confusion matrices, and loss curves.
- **Deep Learning Comparison:** Compare the MLP model with a deep learning model based on the CLIP architecture.

## Getting Started

## Requirements

The following Python libraries are required for this project:

- [numpy~=2.0.1](https://pypi.org/project/numpy/2.0.1/)
- [matplotlib~=3.9.1](https://pypi.org/project/matplotlib/3.9.1/)
- [torch~=2.4.0](https://pypi.org/project/torch/2.4.0/)
- [scikit-learn~=1.5.1](https://pypi.org/project/scikit-learn/1.5.1/)
- [pillow~=10.4.0](https://pypi.org/project/Pillow/10.4.0/)
- [tqdm~=4.66.5](https://pypi.org/project/tqdm/4.66.5/)
- [transformers~=4.44.2](https://pypi.org/project/transformers/4.44.2/)

You can install all of these dependencies using the `requirements.txt` file provided in this repository:

```bash
pip install -r requirements.txt

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danielpan01/PlantDiseaseML.git
   cd PlantDiseaseML

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have suggestions or find any bugs.

## Acknowledgments
Special thanks to Efthimios Gianitsos for his guidance and support in training the models and reviewing the research.
