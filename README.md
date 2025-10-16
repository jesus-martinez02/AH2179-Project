# AH2179-Project
Final Traffic Project for course AH2179 Applied Artificial Intelligence in Transportation at KTH

The structure of the code is the following: 
There are two .py files with the main functionality of processing the datasets and providing the framework to train the model. These are:
- dataset_processing.py: It has the functionallity to do all the preprocessing and addition of the features to the training or test dataset
- models.py: Module that has all the defined models. 


There are also several notebooks that deal with the different specific parts of the project (data analysis, hyperparameter tunning, model evaluation and model diagnostics). These are:
- data_analysis.ipynb
- hyperparameter_tunning.ipynb
- model_evaluation.ipynb
- moddel_diagnostics.ipynb

Additionally, there is a folder nn_models, that contains the keras saved models for the different Neural Networks that have been trained for experiments 1,2 and 3. Each of them is saved in the appropriate subfolder and labelled as nn_model_i.keras, where i represents the iteration number in the respective experiment. Note that for the other models, since the training is deterministic and fast, there is no need to save them in files. Finally, the weather dataset that was fetched from OpenMeteo is included (although it is not used for the final models).

## Installation and starting up
The project has a requirements.txt file containing the library versions that were used. For more information about how to set up a Python environment to work with requirements.txt, go to:
https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
