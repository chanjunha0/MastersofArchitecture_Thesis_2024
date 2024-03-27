"""
Code Structure:
1. Load graph data objects
2. Load or initialize the model
3. Load the number of labeled nodes for each dataset from a CSV
4. Train the model
5. Test the model 
"""

import os
import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from math import sqrt
import matplotlib.pyplot as plt
import time
import csv
from pytorch_models import SimpleGCN, SimpleGAT, SimpleEdgeCNN
import torch.optim as optim

# Set folder path for pytorch data objects
folder_path = r"data\torch_data_object_training"

# Path to the CSV containing labeled nodes info
labels_info_path = r"data\torch_data_object_training\labels_info.csv"

model_path = r"data\1_pytorch_model\model.pth"

# File path to save losses
losses_file_path = r"data\training_and_test_losses.csv"

# Specify model type
model_type = (
    SimpleGCN  # Choose from 'SimpleGCN', 'SimpleGAT', 'SimpleEdgeCNN', 'PointGNNConv',
)


# Placeholder value for unlabeled nodes - adjust as necessary.
placeholder_value = -1


## Functions


def load_graph_data_objects(folder_path, ranges):
    graph_data_objects_by_town = []  # List to hold groups of simulations by town

    for start_run, end_run in ranges:
        town_simulations = []  # List to hold simulations for the current town

        # Load each simulation for the current town based on the specified range
        for run_number in range(start_run, end_run + 1):
            expected_file = f"run_{run_number}.pt"
            graph_path = os.path.join(folder_path, expected_file)
            if os.path.exists(graph_path):
                graph_data = torch.load(graph_path)
                if isinstance(graph_data, Data):
                    town_simulations.append(graph_data)
            else:
                print(f"File does not exist: {graph_path}")

        # Add the current town's simulations to the main list
        graph_data_objects_by_town.append(town_simulations)

    return graph_data_objects_by_town


# Function to load or initialize the model
def load_or_initialize_model(model_path, model_class, *model_args, **model_kwargs):
    if os.path.exists(model_path):
        model = model_class(*model_args, **model_kwargs)
        model.load_state_dict(torch.load(model_path))
        print("Loaded previously trained model.")
    else:
        model = model_class(*model_args, **model_kwargs)
        print("Initialized new model.")
    return model


# Function to load the number of labeled nodes for each dataset from a CSV
def load_num_labeled_nodes(labels_info_path):
    labels_info = pd.read_csv(
        labels_info_path, usecols=[0]
    )  # Load only the first column
    return labels_info.iloc[
        :, 0
    ].tolist()  # Extract values from the first column and convert to list


# Function to Count Graph Data Objects in a Folder
def count_graph_data_objects(folder_path):
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pt"):  # Assuming saved PyTorch tensor files
            count += 1
    return count


def train(model, optimizer, batch):
    model.train()
    optimizer.zero_grad()

    out = model(batch.x, batch.edge_index)
    out = out.squeeze()  # Full model output

    # Here, ensure 'sensor_out' already represents the correct subset of sensors.
    # This might involve adjusting how you generate 'sensor_out' to match the intended subset.
    sensor_out = out[: batch.train_mask.sum().item()]

    # Since 'sensor_out' now directly corresponds to the training sensors,
    # we might not need a mask for indexing, but if we still do:
    # Adjust 'train_mask' to match the size of 'sensor_out' or use it directly if already matched.
    # This step depends on how 'sensor_out' is defined and used.
    valid_out = sensor_out  # If sensor_out is already filtered, no need for further masking here.
    valid_labels = batch.y[batch.train_mask]

    loss = F.mse_loss(valid_out, valid_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, batch, placeholder_value):
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index)
        out = out.squeeze()

        # Apply the test mask to select the output and labels for the test nodes
        test_mask = batch.test_mask
        valid_out = out[test_mask]
        valid_labels = batch.y[test_mask]

        test_loss = F.mse_loss(valid_out, valid_labels).item()

    return test_loss


# Record Start Time
start_time = time.time()
print(start_time)

# SPECIFY THE RANGE OF PT HERE
ranges = [
    (1, 18),
    (19, 36),
    (37, 55),
    (56, 72),
    (73, 91),
    (92, 109),
    (110, 127),
    (128, 145),
    (146, 163),
    (164, 180),
]

# Load graph data objects
graph_data_objects = load_graph_data_objects(folder_path, ranges)

print(f"Loaded {len(graph_data_objects)} sets of graph data objects.")


# Load the number of labeled nodes for each graph
num_labeled_nodes_list = load_num_labeled_nodes(labels_info_path)

# Model Settings
# Ensure at least one town has loaded simulations and each town has at least one simulation
if graph_data_objects and graph_data_objects[0]:
    num_node_features = graph_data_objects[0][
        0
    ].num_node_features  # Access the first simulation of the first town
    print(f"Number of node features: {num_node_features}")
else:
    print("Error: No graph data objects loaded.")

num_classes = 1  # Regression task

# Adjust the criterion according to your task
criterion = torch.nn.MSELoss()

# Specifiy Pytorch Model Type
# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_or_initialize_model(
    model_path, model_type, num_node_features, num_classes
).to(device)
print(f"Model {model_type.__name__} is ready for training.")
if device.type == "cuda":
    print(f"Model is using GPU: {torch.cuda.get_device_name(device)}")
else:
    print("Model is using CPU")


# Using Adam optimizer, lr = learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)


extended_test_mask = None


## Training Loop

train_losses, test_losses, average_epoch_losses = [], [], []

for epoch in range(100):  # Example epoch count
    epoch_loss = 0
    for i, data_list in enumerate(graph_data_objects):
        n_sensors = num_labeled_nodes_list[
            i
        ]  # Assuming a corresponding count for each sublist in graph_data_objects
        for data in data_list:
            if isinstance(data, Data):
                batch_data = data.to(
                    device
                )  # Move each Data object in the sublist to the appropriate device
                train_loss = train(
                    model,
                    optimizer,
                    batch_data,
                )
                epoch_loss += train_loss
    average_loss = epoch_loss / len(graph_data_objects)
    average_epoch_losses.append(average_loss)  # Append the average loss for this epoch
    print(f"Epoch {epoch+1}, Average Loss: {average_loss}")


## Print Test Loss
if extended_test_mask is not None:
    test_loss = test(model, data, extended_test_mask)
    print(f"Test Loss: {test_loss:.2f}")

# After the training loop, plot the average loss per epoch
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), average_epoch_losses, marker="o", linestyle="-", color="b")
plt.title("Average Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.show()

## Save the Model
file_path = rf"data\1_pytorch_model\model.pth"

# Save the model
torch.save(model.state_dict(), file_path)

# Record End Time
end_time = time.time()

# Calculate Training Duration
training_duration = end_time - start_time
minutes, seconds = divmod(training_duration, 60)
hours, minutes = divmod(minutes, 60)
print(
    f"Training took {int(hours)} hours, {int(minutes)} minutes, and {seconds} seconds."
)

# Save the loss values to a CSV file
with open(losses_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Test Loss"])
    for epoch in range(100):
        writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch]])

print("Losses saved.")
