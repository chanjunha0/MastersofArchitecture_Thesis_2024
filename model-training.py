import os
import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from math import sqrt
import matplotlib.pyplot as plt
import time

# Set folder path for pytorch data objects
folder_path = r'data\torch_data_object_training'

# Path to the CSV containing labeled nodes info
labels_info_path = r'data\torch_data_object_training\labels_info.csv' 

model_path = r'data\1_pytorch_model\model.pth' 


## Functions

def load_graph_data_objects(folder_path, start_run, end_run):
    """
    Load graph data objects from .pt files within a specified run number range from a folder.
    
    Args:
    - folder_path (str): The path to the folder containing the graph data files.
    - start_run (int): The start of the run number range.
    - end_run (int): The end of the run number range.
    
    Returns:
    - list: A list of graph data objects loaded from the files.
    """
    graph_data_objects = []
    # Generate expected file names based on the range
    expected_files = [f"run_{i}.pt" for i in range(start_run, end_run + 1)]
    
    for expected_file in expected_files:
        graph_path = os.path.join(folder_path, expected_file)
        if os.path.exists(graph_path):
            print(f"Loading graph data from: {graph_path}")
            graph_data = torch.load(graph_path)
            if isinstance(graph_data, Data):
                graph_data_objects.append(graph_data)
        else:
            print(f"File does not exist: {graph_path}")

    return graph_data_objects

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
    labels_info = pd.read_csv(labels_info_path)
    return labels_info['num_labeled_nodes'].tolist()

# Function to Count Graph Data Objects in a Folder
def count_graph_data_objects(folder_path):
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pt'):  # Assuming saved PyTorch tensor files
            count += 1
    return count

def train(model, optimizer, data, mask):
    model.train()
    optimizer.zero_grad()
    
    out = model(data)  # Model outputs for all nodes
    out = out.squeeze()
    
    # Mask to identify non-placeholder labels
    # Adjust this condition based on your placeholder_value and task
    valid_label_mask = data.y != placeholder_value
    
    # Apply both the extended mask and valid_label_mask to filter predictions and labels
    valid_out = out[mask & valid_label_mask]
    valid_labels = data.y[mask & valid_label_mask]
    
    loss = F.mse_loss(valid_out, valid_labels)  # Calculate loss on non-placeholder, valid data points only
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def test(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        out = out.squeeze()  # Ensure the output matches expected dimensions

        # Ensure the mask is for the full set of nodes, adjusting if it was originally for labeled nodes only
        if mask.size(0) != out.size(0):
            # This assumes the original mask size corresponds to the number of labeled nodes
            # and needs to be extended to match the full set of nodes (similar to what was done for training)
            extended_mask = torch.cat((mask, torch.zeros(data.num_nodes - mask.size(0), dtype=torch.bool, device=mask.device)))
        else:
            extended_mask = mask

        # Apply the extended mask to filter predictions and labels for loss calculation
        valid_out = out[extended_mask]
        valid_labels = data.y[extended_mask]
        
        test_loss = F.mse_loss(valid_out, valid_labels).item()

    return test_loss






## Classes

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        '''
        Initialises a model that processes graph-structured data using 
        simple two-layer GCN architecture.
        
        Args:
        num_node_features = number of features each node in the input graph has
        num_classes = number of output classes
        '''
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)  # First GCN layer
        self.conv2 = GCNConv(16, num_classes)        # Second GCN layer

    def forward(self, data):
        '''
        Defines the forward pass of the model

        Args:
        data = object containing graph data
        '''
        x, edge_index = data.x, data.edge_index    # Extract node features and graph structure
        x = self.conv1(x, edge_index)    # Apply the first GCN convolution layer
        x = F.relu(x)  # Apply the ReLU activation function to the input tensor x
        x = F.dropout(x, training=self.training) # Apply dropout regularisation to reduce overfitting during training
        x = self.conv2(x, edge_index) # Apply the second GCN convolution layer
        
        return x  # model's predictions for each node 
    







# Record Start Time
start_time = time.time()

# SPECIFY THE RANGE OF PT HERE
start_run, end_run = 1,54   

# Load graph data objects
graph_data_objects = load_graph_data_objects(folder_path, start_run=start_run, end_run=end_run)

print(f'Loaded {len(graph_data_objects)} graph data objects.')

# Count graph data objects directly from the folder 

num_graphs = count_graph_data_objects(folder_path)
print(f'Total number of graph data objects in the folder: {num_graphs}')

# Load the number of labeled nodes for each graph
num_labeled_nodes_list = load_num_labeled_nodes(labels_info_path)  

# Model Settings
num_node_features = graph_data_objects[0].num_node_features  #  Retrieves the number of features associated with the nodes
num_classes = 1  # Set the number of output classes (no changes needed)

# Adjust the criterion according to your task
criterion = torch.nn.MSELoss()  

model = load_or_initialize_model(model_path, SimpleGCN, num_node_features, num_classes)

# Using Adam optimizer, lr = learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Check if GPU is avaliable 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
if device.type == 'cuda':
    print(f'Model is using GPU: {torch.cuda.get_device_name(device)}')
else:
    print('Model is using CPU')



## Training Loop

train_losses = []
test_losses = []

for idx, data in enumerate(graph_data_objects):
    try:  # Start of the try block
        data = data.to(device)  # Move data to the appropriate device
        
        num_labeled_nodes = num_labeled_nodes_list[idx]  # Retrieve the number of labeled nodes for the current graph
        num_total_nodes = data.num_nodes  # Total number of nodes in the current graph

        # Placeholder value for unlabeled nodes - choose based on your task
        placeholder_value = 0

        # Extend data.y with the placeholder value for unlabeled nodes
        if data.y.size(0) < num_total_nodes:
            # Calculate the number of unlabeled nodes
            num_unlabeled_nodes = num_total_nodes - data.y.size(0)
            
            # Create a tensor of placeholders for unlabeled nodes
            placeholders = torch.full((num_unlabeled_nodes,), placeholder_value, dtype=data.y.dtype)
            
            # Extend data.y
            data.y = torch.cat((data.y, placeholders))

        # Generate initial masks for labeled nodes only
        train_mask = torch.zeros(num_labeled_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_labeled_nodes, dtype=torch.bool)

        # Randomly permute indices of labeled nodes to split into training and testing
        indices = torch.randperm(num_labeled_nodes)
        num_train = int(num_labeled_nodes * 0.8)  # Split, e.g., 80% for training

        # Set training and testing masks for labeled nodes
        train_mask[indices[:num_train]] = True
        test_mask[indices[num_train:]] = True

        # Extend masks to cover all nodes
        extended_train_mask = torch.cat((train_mask, torch.ones(num_total_nodes - num_labeled_nodes, dtype=torch.bool)))
        extended_test_mask = torch.cat((test_mask, torch.zeros(num_total_nodes - num_labeled_nodes, dtype=torch.bool)))

        # Training and testing loop
        for epoch in range(100):  # Adjust the number of epochs as needed
            try:  # Nested try for the inner loop
                train_loss = train(model, optimizer, data, extended_train_mask)  # Assuming extended masks are used
                test_loss = test(model, data, extended_test_mask)
                
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                
                print(f'Graph {idx+1}, Epoch {epoch+1}: Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
            except RuntimeError as e:  # Catch specific runtime errors
                print(f"Encountered an error in Graph {idx+1}, Epoch {epoch+1}: {e}. Skipping to the next epoch.")
                continue  # Skip to the next epoch

    except Exception as e:  # Catch other exceptions
        print(f"Encountered an error processing Graph {idx+1}: {e}. Skipping to the next graph.")
        continue  # Skip to the next graph in the outer loop






## Print Test Loss
test_loss = test(model, data, test_mask)
print(f'Test Loss: {test_loss:.2f}')

## Save the Model
file_path = rf'data\1_pytorch_model\model.pth'  

# Save the model
torch.save(model.state_dict(), file_path)

# Record End Time
end_time = time.time()

# Calculate Training Duration
training_duration = end_time - start_time
minutes, seconds = divmod(training_duration, 60)
hours, minutes = divmod(minutes, 60)
print(f"Training took {int(hours)} hours, {int(minutes)} minutes, and {seconds} seconds.")

## Visualise Loss function
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.show()