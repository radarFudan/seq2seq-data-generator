Examples and descriptions are in description.ipynb
To run the notebook and data_generator need to install requirements.txt which basically includes numpy and matplotlib.

To run codes relates to the model need to install requirements_full.txt which includes pytorch and pytorch lightning.

For database generation, use database_generator.py. The parameters of the database are defined in batch_gen_util.py. A memory calculator function to calculate the memory given a kernel is included. All parameters and memory information will be written to a .txt file.

#### Update from shida

seq2seq-data-generator is at the same level w.r.t. 3W.

-seq2seq-data-generator

    - data: Stored process data, in the format of x/y _ strain/test

-3W

#### Potential improvement

Try other sequence model, CNN / Transformer, maybe we can use linear combination of these models to learn a hybrid model, and then check the memory of the hybrid model.



1. Synthetic dataset
2. Hyperparameter tuning
3. Mori-Danzig Formalism ...
