# Identify the memory pattern in the real world sequence datasets

## How to run

Examples and descriptions are in `description.ipynb`.

To run the notebook and data_generator need to install `requirements.txt` which basically includes numpy and matplotlib.

- `pip install -r requirements.txt`

To run codes relates to the model need to install `requirements_full.txt` which includes pytorch and pytorch lightning.

- `pip install -r requirements_full.txt`

For database generation, use `database_generator.py`. The parameters of the database are defined in `batch_gen_util.py`. 
A memory calculator function to calculate the memory given a kernel is included. All parameters and memory information will be written to a `.txt` file.

## Real world dataset

1. Synthetic dataset - in Data folder
2. TCW - total column water 
3. 3W (https://github.com/petrobras/3W.git)

    Classification task, to evaluate the accuracy

    Put the seq2seq-data-generator repo in the same folder as 3W.
4. CFD

    Regression task, 

5. \* Oil temperature

### Future improvement

Try other sequence model, CNN / Transformer, maybe we can use linear combination of these models to learn a hybrid model, and then check the memory of the hybrid model.


1. Synthetic dataset
    Linear functional with different memory patterns:
    - Exp
    - Pol
    - Two parts (different memory length)
    - Airy
    - Shift memory

    The memory has been defined in `batch_gen_util.py`

2. Hyperparameter tuning
    Ray
    What to tune?
    - Hidden dimension [32, 256], uniform
    - Learning rate [0.001, 0.1], uniform
    - Layers [1, 4], uniform

3. Mori-Zwanzig formalism
    - https://en.wikipedia.org/wiki/Mori-Zwanzig_formalism




