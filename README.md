<!-- SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH. -->
<!-- SPDX-License-Identifier: MIT -->
# neural_representation_of_differentiable_trees

This repository contains the implementation code for the paper "NeRDT: Neural Representation of Differentiable Trees for Fast and Interpretable Inference"
Please note: This is an archived project, thus it is not actively maintained. Contributing is not endorsed.

Author: Tobias Ritter <tobias.ritter@mercedes-benz.com>, on behalf of MBition GmbH.

Source code has been tested solely for our own use cases, which might differ from yours.

[Provider Information](https://github.com/mercedes-benz/foss/blob/master/PROVIDER_INFORMATION.md)

## Cloning the Source Code

In order for all experiments to run, this repository relies on git submodules to include the source code of reference models. The following command will clone this repository as well as the required submodules:

`git clone --recurse-submodules <url>`

## Package Installation

NeRDT requires Python `3.10.11` and can be installed as a Python package after cloning the repository as described above:

```bash
cd nerdt
pip install .
```

## Repository Structure

+ src: contains models, data preprocessing and all other functions
    + abstract: contains model wrapper classes
    + data: contains the data loader classes
    + export: contains code for logging and exporting results to SQLite
    + models: contains all model implementations
    + utils: contains utility functions
    + validation: code for evaluating models, hyperparameter tuning, benchmarking etc.
+ test: unit tests for selected functions

## Running the Experiments

To be able to run all experiments described in our paper, it is first required to install TEL as a reference model.
The installation instructions for TEL can be found [here](https://github.com/google-research/google-research/blob/master/tf_trees/README.md). All results were achived in a Python `3.10.11` environment. In case you did not install NeRDT as a package, you can install only its requirements instead:

```bash
pip install -r requirements.txt
```

Furthermore, the experiments expect the following data sets to be located at the relative path `./data`:
+ [Abalone](https://archive.ics.uci.edu/dataset/1/abalone) (Save under `./data/abalone.data`)
+ [MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg) (Save under `./data/auto-mpg.data`)
+ [EE](https://archive.ics.uci.edu/dataset/242/energy+efficiency) (Save under `./data/ENB2012_data.xlsx`)
+ [News](https://archive.ics.uci.edu/dataset/332/online+news+popularity) (Save under `./data/OnlineNewsPopularity.csv`)

There are a total of 7 experiments, which can be run as follows:

+ tuning.py: `python tuning.py <dataset> <model>` - e.g. `python tuning.py mpg nerdt`
+ timing.py: `python timing.py <dataset> <model> <depth>` - e.g. `python timing.py mpg nerdt 10` 
+ pruning_accuracy.py: `python pruning_accuracy.py <dataset>` - e.g. `python pruning_accuracy.py mpg`
+ pruning_timing.py: `python pruning_timing.py <dataset> <depth>` - e.g. `python pruning_timing.py mpg 5`
+ pruning_timing_ref.py: `python pruning_timing_ref.py <dataset> <depth>` - e.g. `python pruning_timing_ref.py mpg 5`
+ forest_tuning.py: `forest_tuning.py <dataset>` - e.g. `python forest_tuning.py mpg`
+ forest_timing.py: `python forest_timing.py <dataset>` - e.g. `python forest_timing.py mpg`

## How to Use NeRDT as a One Layer Of Your Model

The `NerdtLayer` can be found in `src/models/nerdt_lib/layers.py` and can be used like other keras layers:

```python
import tensorflow as tf

model = tf.keras.Sequential(
    layers=[
        ..., 
        NerdtLayer(depth=5, activation=tf.math.sigmoid), 
        ...,
    ]
)
```

## Citing NeRDT

If you find this work useful in your research, please consider citing the following paper:

```
@article{
    ...
}
```
