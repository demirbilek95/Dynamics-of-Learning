# The Dynamics of Learning Beyond Stochastic Gradient Descent

This repository contains the code used to obtain the experiment results of the thesis (the title is above). For the writing of the thesis, please check [this repository](https://github.com/demirbilek95/oxforddown) (not completed yet).
All of the experiments are stochastic. Setting a seed may influence the performance because the training phase takes benefit of this stochasticity. However, the results should be around the vicinity of the ones reported in the writings.

## Structure of the Repository

* [requirements.txt](requirements.txt) contains the modules that are used with their versions.
* [plots](plots) contains the figures that are produced from the experiments.
* [runs](runs) contains the accuracy and loss values from the various experiments.
* [scripts](scripts) includes the python files that are used from the notebooks.
	* [architecture](scripts/architecture.py) includes the neural network architectures that are used in the experiments.
	* [data](scripts/data.py) contains the code to produce the `MNIST parity` data and `Random Data`.
	* [notebook_utils](scripts/notebook_utils.py) and [plot_utils](scripts/plot_utils.py) includes the helper functions that are used in notebooks.
	* [optimizer](scripts/optimizer.py) includes the various optimizer algorithms that are used by the networks.
	* [train](scripts/train.py) and [train_utils](scripts/train_utils.py) includes the reusable functions to train and test the networks.
* [01-Parity Experiments](01-Parity_Experiments.ipynb) contains the code that are used to produce `MNIST Parity` Experiments.
* [02-Random Data Experiments](02-Random_Data_Experiments.ipynb) contains the code that are used to produce `Random Data` Experiments.
