import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_models import SimpleGCN, SimpleGAT, SimpleEdgeCNN

# model_paths = ["bedok", "bishan", "bukit_merah", "jurong_west", "jurong_east"]
model_paths = ["bukit_merah"]
object = "run_1"
num_node_features = 4
num_classes = 1

# Load your data object for prediction
new_data = torch.load(f"data/torch_data_object_prediction/{object}.pt")
if isinstance(new_data, Data):
    new_data = [new_data]
data_loader = DataLoader(new_data, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_path in model_paths:
    model = SimpleGCN(num_node_features, num_classes)
    model.load_state_dict(torch.load(f"data/1_pytorch_model/{model_path}.pth"))
    model.eval()
    model.to(device)

    predictions = []
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.edge_index).squeeze()
            # Slice the output to consider only indices [0, 720]
            sliced_output = output[:720].cpu().numpy().flatten().tolist()
            predictions.extend(sliced_output)

    # Save each model's predictions to a CSV file
    prediction_file_path = (
        f"data/2_prediction_testing/{model_path}_predictions_{object}.csv"
    )
    np.savetxt(prediction_file_path, np.array(predictions), delimiter=",")
    print(f"Saved predictions for {model_path} to: {prediction_file_path}")


dfs = []
for model_path in model_paths:
    df = pd.read_csv(
        f"data/2_prediction_testing/{model_path}_predictions_{object}.csv", header=None
    )
    dfs.append(df)

# Assuming all DataFrames have the same structure, average them
avg_predictions = pd.concat(dfs, axis=1).mean(axis=1)

# Save the averaged predictions
avg_file_path = f"data/2_prediction_testing/averaged_predictions_{object}.csv"
avg_predictions.to_csv(avg_file_path, index=False, header=False)
print(f"Saved averaged predictions to: {avg_file_path}")
