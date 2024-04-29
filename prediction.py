"""
This script is used to predict the output of a trained model on a new dataset.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_models import SimplifiedGCN

model_path = "model_72"
num_node_features = 16
num_classes = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV to get 'n' values for each run, assuming headerless CSV
labels_info_path = r"C:\Users\colin\OneDrive\Desktop\Thesis Part 2\thesis - simulation\data\torch_data_object_prediction\labels_info.csv"
labels_info_df = pd.read_csv(labels_info_path, header=None, names=["run_name"])
print(labels_info_df)

model = SimplifiedGCN(num_node_features, num_classes)
model.load_state_dict(torch.load(f"data/1_pytorch_model/{model_path}.pth"))
model.eval()
model.to(device)

object_start = 1
object_end = 9

for object_num in range(object_start, object_end + 1):
    object_name = f"run_{object_num}"

    # Assuming rows in CSV are in the same order as object_nums
    n_results = int(labels_info_df.iloc[object_num - 1]["run_name"])  # Cast to int here

    new_data = torch.load(f"data/torch_data_object_prediction/{object_name}.pt")
    if isinstance(new_data, Data):
        new_data = [new_data]
    data_loader = DataLoader(new_data, batch_size=1, shuffle=False)

    predictions = []
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.edge_index).squeeze()
            output = output.cpu().numpy().flatten()
            predictions.extend(output.tolist())

    # Slice the predictions to keep only the first 'n' results
    predictions = predictions[:n_results]

    # Save the sliced predictions for each object
    prediction_file_path = (
        f"data/2_prediction_testing/{model_path}_predictions_{object_name}.csv"
    )
    np.savetxt(prediction_file_path, np.array(predictions), delimiter=",")
    print(f"Saved predictions for {model_path} to: {prediction_file_path}")


# # Function to reverse min-max scaling
# def reverse_min_max_scaling(scaled_val, min_val, max_val):
#     """
#     Reverses the min-max scaling for a value or array of values.

#     Args:
#     - scaled_val: The scaled value(s) to be reversed.
#     - min_val: The original minimum value used for scaling.
#     - max_val: The original maximum value used for scaling.

#     Returns:
#     - The original value(s) before scaling.
#     """
#     return (scaled_val * (max_val - min_val)) + min_val

# # Load normalization parameters
# norm_params_path = "data/config/global_normalization_parameters.json"
# with open(norm_params_path, "r") as f:
#     normalization_params = json.load(f)

# # Correctly reference 'label' for normalization parameters
# min_val, max_val = (
#     normalization_params["label"]["min"],
#     normalization_params["label"]["max"],
# )
