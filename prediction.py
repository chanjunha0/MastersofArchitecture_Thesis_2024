"""
This script is used to predict the output of a trained model on a new dataset.
"""

# reminder to normalise the prediction data objects


import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_models import SimpleGCN  # Assuming SimpleGCN is the model you're using
import json  # For loading normalization parameters


# Function to reverse min-max scaling
def reverse_min_max_scaling(scaled_val, min_val, max_val):
    """
    Reverses the min-max scaling for a value or array of values.

    Args:
    - scaled_val: The scaled value(s) to be reversed.
    - min_val: The original minimum value used for scaling.
    - max_val: The original maximum value used for scaling.

    Returns:
    - The original value(s) before scaling.
    """
    return (scaled_val * (max_val - min_val)) + min_val


model_path = "model_normalised_gcn"  # Specify the model to use
object = "run_3"
num_node_features = 4
num_classes = 1

# Load normalization parameters
norm_params_path = "data/config/global_normalization_parameters.json"
with open(norm_params_path, "r") as f:
    normalization_params = json.load(f)

# Correctly reference 'label' for normalization parameters
min_val, max_val = (
    normalization_params["label"]["min"],
    normalization_params["label"]["max"],
)

# Load your data object for prediction
new_data = torch.load(f"data/torch_data_object_prediction/{object}.pt")
if isinstance(new_data, Data):
    new_data = [new_data]  # Ensure new_data is a list for DataLoader
data_loader = DataLoader(new_data, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the model
model = SimpleGCN(num_node_features, num_classes)
model.load_state_dict(torch.load(f"data/1_pytorch_model/{model_path}.pth"))
model.eval()
model.to(device)

predictions = []
for data in data_loader:
    data = data.to(device)
    with torch.no_grad():
        output = model(data.x, data.edge_index).squeeze()
        # Assume the output needs no slicing here; adjust as necessary
        scaled_output = (
            output.cpu().numpy().flatten()
        )  # Convert to numpy array and flatten
        # Reverse normalize the predictions
        reversed_predictions = reverse_min_max_scaling(scaled_output, min_val, max_val)
        predictions.extend(reversed_predictions.tolist())

# Save the reverse normalized model's predictions to a CSV file
prediction_file_path = (
    f"data/2_prediction_testing/{model_path}_predictions_{object}.csv"
)
np.savetxt(prediction_file_path, np.array(predictions), delimiter=",")
print(
    f"Saved reverse normalized predictions for {model_path} to: {prediction_file_path}"
)
