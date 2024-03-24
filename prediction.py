import numpy as np
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.data import Data
from pytorch_models import SimpleGCN, SimpleGAT, SimpleEdgeCNN

# Paste model path
model_iteration = "model_gcn_3L_180_2_d05"

# Model Class
model_type = SimpleGCN  # Specify model type

# Input Pytorch Object Date
# Copy paste in here
object = "run_3"

# Set the slicing range for the indices (labeled nodes)
sensor_indices = slice(0, 720)

# Path to save script
file_path = rf"data\2_prediction_testing\sensor_predictions_{object}.csv"

num_node_features = 4
num_classes = 1

model = model_type(num_node_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Load model
model.load_state_dict(torch.load(rf"data\1_pytorch_model\{model_iteration}.pth"))

# Set the model to evaluation mode
model.eval()

# Load your data object for prediction
new_data = torch.load(rf"data\torch_data_object_prediction\{object}.pt")

print("Data object loaded")

# If 'new_data' is a single Data object, wrap it in a list for DataLoader compatibility
if isinstance(new_data, Data):
    new_data = [new_data]

# Create a DataLoader for your new dataset
data_loader = DataLoader(new_data, batch_size=1, shuffle=False)

# Check if GPU is avaliable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda":
    print(f"Model is using GPU: {torch.cuda.get_device_name(device)}")
else:
    print("Model is using CPU")

# Make predictions
predictions = []
for data in data_loader:
    data = data.to(device)
    with torch.no_grad():
        output = model(
            data.x, data.edge_index
        )  # Simplified and correct for all model types
        output = output.squeeze()  # Ensure this line uses 'output', not 'out'
        sensor_output = output[sensor_indices]  # Adjusted slicing
        predictions.append(sensor_output.cpu().numpy())

# Display the shape of predictions
predictions_array = np.array(predictions)
print("Shape of predictions array:", predictions_array.shape)

print("Output shape:", output.shape)
print("Sensor output shape:", sensor_output.shape)

# Reshape predictions_array to have shape (592, 1)
predictions_array = np.vstack(predictions)
predictions_array = predictions_array.T
print("Shape of sensor predictions array:", predictions_array.shape)

# Export rounded predictions to CSV in the same folder as the script
np.savetxt(file_path, predictions_array, delimiter=",")
print(f"Saved predictions to: {file_path}")
