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
from pytorch_models import (
    SimpleGCN,
    SimpleGAT,
    SimpleEdgeCNN,
    ComplexGCN,
    SimplifiedGCN,
)
import torch.optim as optim
import datetime

# Set folder path for pytorch data objects
folder_path = r"data\torch_data_object_training"

# Path to the CSV containing labeled nodes info
labels_info_path = r"data\torch_data_object_training\labels_info.csv"

model_path = r"data\1_pytorch_model\model.pth"

# File path to save losses
losses_file_path = r"data\training_and_test_losses.csv"

# Specify model type
model_type = SimplifiedGCN


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


def test(model, batch):
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index)
        out = out.squeeze()  # Full model output

        # Assuming labeled nodes correspond to the first 'N' entries in your labels 'y'
        # and 'out' tensor has predictions for all nodes.
        # You need to align 'out' with the labeled nodes before applying the mask.
        # This example assumes the labeled nodes are the first ones matching the length of 'y'.
        labeled_out = out[: len(batch.y)]

        # Now, apply the test mask to select the output for the test nodes
        test_out = labeled_out[batch.test_mask]
        valid_labels = batch.y[batch.test_mask]

        test_loss = F.mse_loss(test_out, valid_labels).item()

    return test_loss


# Record Start Time
start_time = time.time()
print(start_time)

# SPECIFY THE RANGE OF PT HERE
# ranges = [
#     (1, 18),
#     (19, 36),
#     (37, 55),
#     (56, 72),
#     (73, 91),
#     (92, 109),
#     (110, 127),
#     (128, 145),
#     (146, 163),
#     (164, 180),
#     (181, 198),
#     (199, 216),
# ]

total_objects = 72

# Generate ranges to load each object one by one
ranges = [(i, i) for i in range(1, total_objects + 1)]

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
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)


extended_test_mask = None


## Training and Testing Loop

train_losses, test_losses, average_epoch_losses = [], [], []

# Record Start Time before the loop
total_start_time = time.time()

for epoch in range(20):  # Example epoch count
    # Record Start Time for the current epoch
    epoch_start_time = time.time()

    epoch_loss = 0
    for (
        data_list
    ) in (
        graph_data_objects
    ):  # 'graph_data_objects' is a list of lists of 'Data' objects
        for data in data_list:  # Iterate through each 'Data' object
            data = data.to(device)  # Move the 'Data' object to the appropriate device
            train_loss = train(model, optimizer, data)
            epoch_loss += train_loss
    average_loss = epoch_loss / sum(len(data_list) for data_list in graph_data_objects)
    average_epoch_losses.append(average_loss)

    # Calculate test loss at the end of the epoch
    model.eval()
    test_epoch_loss = 0
    with torch.no_grad():
        for data_list in graph_data_objects:  # Reuse 'graph_data_objects' for testing
            for data in data_list:  # Iterate through each 'Data' object for testing
                data = data.to(device)  # Ensure data is on the correct device
                test_loss = test(
                    model, data
                )  # Assumes your 'test' function uses 'test_mask'
                test_epoch_loss += test_loss
    average_test_loss = test_epoch_loss / sum(
        len(data_list) for data_list in graph_data_objects
    )
    test_losses.append(average_test_loss)

    # Calculate and print runtime for the epoch
    epoch_runtime = time.time() - epoch_start_time

    # Calculate total time elapsed
    total_elapsed_time = time.time() - total_start_time

    print(
        f"Epoch {epoch+1}, Average Train Loss: {average_loss:.2f}, Average Test Loss: {average_test_loss:.2f}, Runtime: {epoch_runtime:.2f} seconds, Total Elapsed Time: {total_elapsed_time:.2f} seconds"
    )


# Plotting
plt.figure(figsize=(10, 6))

# Plotting the average training loss per epoch.
# 'average_epoch_losses' contains the average loss for each epoch during the training phase.
plt.plot(
    range(
        1, len(average_epoch_losses) + 1
    ),  # Ensure x-axis matches the number of epochs recorded
    average_epoch_losses,
    marker="o",
    linestyle="-",
    color="b",
    label="Training Loss",
)

# Plotting the average test loss per epoch.
# 'test_losses' contains the average loss for each epoch during the testing phase.
# Note: It's important that the length of 'test_losses' matches 'average_epoch_losses'.
plt.plot(
    range(
        1, len(test_losses) + 1
    ),  # Ensure x-axis matches the number of epochs recorded for consistency
    test_losses,
    marker="s",
    linestyle="--",
    color="r",
    label="Test Loss",
)


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

# Get the current date and time
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Update the CSV file name to include the timestamp
losses_file_path = f"data/training_and_test_losses_{timestamp}.csv"

# Save the train_losses and test_losses to the CSV file
with open(losses_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Training Loss", "Test Loss"])
    for epoch, (train_loss, test_loss) in enumerate(
        zip(average_epoch_losses, test_losses), start=1
    ):
        writer.writerow([epoch, train_loss, test_loss])

print("Losses saved.")

plt.title("Average Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
