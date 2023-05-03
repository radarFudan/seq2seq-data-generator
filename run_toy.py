import os
import time
from pathlib import Path

from libs.train import *

# from libs.train_with_tune import *


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    T = 256

    # memory_type_list = ["Exp", "ExpPeak", "Shift", "TwoPart"]
    memory_type_list = ["Exp", "shift",]


    for memory_type in memory_type_list:
        data_dir = Path(f"./data/{memory_type}")
        if os.path.exists(data_dir):
            index = 0

            with open(f"{data_dir}/in_{index}.pickle", 'rb') as f:
                train = pickle.load(f)
            with open(f"{data_dir}/out_{index}.pickle", 'rb') as f:
                train_output = pickle.load(f)

            train_output = np.load(f"{data_dir}/out_{index}.pickle", allow_pickle=True)

            # Normalization
            train /= np.max(train)
            train_output /= np.max(train_output)

            # with open('filename.pkl', 'rb') as f:
            # # Load the pickled data from the file
            #     data = pickle.load(f)
        else:
            raise NotImplementedError

        # TODO
        # train test split

        activation = "tanh"  # Tanh RNN
        hid_dim = 512
        num_layers = 1
        input_dim = 1
        output_dim = 1
        config = {}
        config["loss"] = torch.nn.MSELoss()
        config["optim"] = "Adam"
        config["lr"] = 0.0001

        dtype = torch.float32

        train_toy(
            f"{activation}RNN_toy",
            RNNModel(
                config=config,
                hid_dim=hid_dim,
                num_layers=num_layers,
                input_dim=input_dim,
                output_dim=output_dim,
                return_sequence=False,
                dtype=32,
            ),
            train,
            train_output,
            train,
            train_output,
            call_backs=[
                EarlyStopping(
                    monitor="valid_loss",
                    min_delta=1e-10,
                    patience=5,
                    verbose=True,
                    mode="min",
                )
            ],
            devices=1,
            dtype=dtype,
        )
