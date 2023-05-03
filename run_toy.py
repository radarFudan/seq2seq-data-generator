import os
import time
from pathlib import Path

from libs.train import *

# from libs.train_with_tune import *


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    T = 256

    memory_type_list = ["Exp", "ExpPeak", "Shift", "TwoPart"]
    memory_type_list_prefix = ["exp", "epk_lambda", "shift", "TP_"]



    for memory_type, memory_type_list in zip(memory_type_list, memory_type_list_prefix):
        data_dir = Path(f"./Data/{memory_type}")
        if os.path.exists(data_dir):
            train = torch.load(f"{data_dir}/in_{}.pickle")
            train_output = torch.load(f"{data_dir}/out_.pickle")

            # Normalization
            train /= torch.max(train)
            train_output /= torch.max(train_output)
        else:
            raise NotImplementedError

        # TODO
        # conduct train test split using keras

        print(train.shape, train_output.shape, test.shape, test_output.shape)
        print(train.dtype, train_output.dtype, test.dtype, test_output.dtype)
        exit()

        activation = "tanh"  # Tanh RNN
        hid_dim = 512
        num_layers = 1
        input_dim = 3
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
            test,
            test_output,
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
