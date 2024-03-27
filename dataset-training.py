"""
Code Structure:
1. Import necessary libraries
2. Define functions to normalize data and process each run
3. Loop over the range of runs and process each run 'process_run'
- load CSV files
- append material properties
- combine building DataFrames
- extract vertex lengths
- map sensors ID to vertices ID
- append distance values to mapped df
- combine sensor and building DataFrames
- prepare node features
- create continuous index for sensors and vertices
- adjust vertex_id in edge_df
- map sensor_id and vertex_id to their respective indices
- filter out any edges that couldn't be mapped
- convert to torch tensor
- ensure data type compatibility
- update labels for sensors with their radiation values
- create torch tensor with compatible data type
- initialize mask tensors with zeros (False)
- example strategy: simple split (e.g., 80% train, 20% test) for labeled nodes
- creating the Data object for training or prediction
4. Print completion message
"""

import os
import pandas as pd
import torch
from torch_geometric.data import Data
import json
import numpy as np
import csv

# Define the mode at the start of your code
pipeline_mode = "training"  # Change to "prediction/training" when needed
norm_params_path = r"C:\Users\colin\OneDrive\Desktop\Thesis Part 2\thesis - simulation\data\6_normalisation"

# # Load normalization parameters
# norm_params = {
#     "building": json.load(
#         open(os.path.join(norm_params_path, "building_norm_params.json"), "r")
#     ),
#     "distance": json.load(
#         open(os.path.join(norm_params_path, "distance_norm_params.json"), "r")
#     ),
#     "label": json.load(
#         open(os.path.join(norm_params_path, "label_norm_params.json"), "r")
#     ),
#     "sensor": json.load(
#         open(os.path.join(norm_params_path, "sensor_norm_params.json"), "r")
#     ),
# }


# # Function to normalize given columns of a DataFrame with specified mean and std
# def normalize_df(df, means, stds):
#     return (df - means) / stds

# Path to the CSV containing labeled nodes info
labels_info_path = r"data\torch_data_object_training\labels_info.csv"


# Load the number of labeled nodes from the first column of the CSV
labels_info = pd.read_csv(labels_info_path, header=None, usecols=[0])
num_labeled_nodes_list = [int(x) for x in labels_info[0].tolist()]


training_mask_log = pd.DataFrame(columns=["Run", "Training Indices"])


def process_run(run_num, num_labeled_nodes):
    file_path_training = rf"data\csv_training\{run_num}"
    filename_training = rf"data\torch_data_object_training\{run_num}.pt"
    filename_prediction = rf"data\torch_data_object_prediction\{run_num}.pt"

    # Csv Files to Import
    file_names = [
        "building_alum_vertex",
        "building_brick_vertex",
        "building_conc_vertex",
        "building_glass_vertex",
        "building_wood_vertex",
        "distance_alum",
        "distance_brick",
        "distance_conc",
        "distance_glass",
        "distance_wood",
        "label",
        "sensor",
        "sensor_length",
        "vertex_length_alum",
        "vertex_length_brick",
        "vertex_length_conc",
        "vertex_length_glass",
        "vertex_length_wood",
    ]

    # Load and normalize CSV data
    dfs = {}  # Store normalized DataFrames here

    # Dictionary mapping DataFrame names to material names
    df_material_map = {
        "building_conc_vertex_df": "exterior_concrete_wall",
        "building_glass_vertex_df": "exterior_glass",
        "building_wood_vertex_df": "exterior_wood_wall",
        "building_brick_vertex_df": "exterior_white_brick",
        "building_alum_vertex_df": "exterior_alum_cladding",
    }

    # List to combine into building_df
    append_list = (
        "building_alum_vertex_df_append",
        "building_brick_vertex_df_append",
        "building_conc_vertex_df_append",
        "building_glass_vertex_df_append",
        "building_wood_vertex_df_append",
    )

    # List of dfs to extract vertex length
    df_vertex_names = [
        "vertex_length_alum_df",
        "vertex_length_brick_df",
        "vertex_length_conc_df",
        "vertex_length_glass_df",
        "vertex_length_wood_df",
    ]

    # Material list
    material_list = ["alum", "brick", "conc", "glass", "wood"]

    def print_df(dataframes, line_length=50):
        """
        Prints the head of each DataFrame in the list, separated by a line.

        Parameters:
        - dataframes (list of tuple): A list where each tuple contains a string (the name of the DataFrame) and a DataFrame.
        - line_length (int): The length of the separating line. Default is 50.
        """
        line = "-" * line_length
        for name, df in dataframes:
            print(name)
            print(df.head())
            print(line)

    def format_and_insert_id_column(df, id_base_name):
        """
        Resets the index of the DataFrame, creates a new column with formatted IDs based on the new index,
        and inserts this new column as the first column of the DataFrame.

        Args:
        - df (pd.DataFrame): The DataFrame to operate on.
        - id_base_name (str): The base name for the new ID column (e.g., 'sensor_id', 'vertex_id').

        Returns:
        - pd.DataFrame: The modified DataFrame with the new ID column as the first column.
        """
        # Resetting the index so the specified ID column is no longer the index column, if it was.
        df.reset_index(drop=True, inplace=True)

        # Creating a new ID column with formatted values.
        df[id_base_name] = [f"{id_base_name}_{i + 1}" for i in df.index]

        # Inserting the new ID column as the first column.
        df.insert(0, id_base_name, df.pop(id_base_name))

        return df

    def load_csv_files_as_dict(file_path_training, file_names):
        """
        Loads CSV files into pandas DataFrames and stores them in a dictionary with dynamically constructed keys.
        Skips empty CSV files.

        Args:
        - file_path_training (str): The directory path where the CSV files are stored.
        - file_names (list of str): List of file names without the '.csv' extension.

        Returns:
        - dict: A dictionary where each key is a dynamically constructed name based on the file name and
                each value is the corresponding DataFrame loaded from the CSV file.
        """
        # Dictionary to store the DataFrames, with keys as dynamically constructed names
        dataframes = {}

        # Loop over the list of file names
        for file_name in file_names:
            file_path = os.path.join(file_path_training, f"{file_name}.csv")
            # Check if the file is not empty (size > 0)
            if os.path.getsize(file_path) > 0:
                # Construct the DataFrame name and load the CSV file into a DataFrame
                df_name = f"{file_name}_df"
                dataframes[df_name] = pd.read_csv(file_path, header=None)
            # else:
            #     # print(f"Skipped empty file: {file_name}.csv")

        return dataframes

    def append_material_properties(dfs, material_df, df_material_map):
        """
        Appends material properties to each row of specified building DataFrames based on a mapping dictionary,
        creating new DataFrames for the appended versions without modifying the originals.
        Skips empty DataFrames and those not found in the dfs dictionary.

        Args:
        - dfs (dict): Dictionary of DataFrames to be updated, where keys are DataFrame names.
        - material_df (pd.DataFrame): DataFrame containing material properties.
        - df_material_map (dict): Dictionary mapping DataFrame names to material names.

        Returns:
        - dict: The dfs dictionary with new DataFrames added that contain the original data
                with material properties appended. The new DataFrames have keys with an `_append` suffix.
        """
        for df_name, material_name in df_material_map.items():
            # Construct new DataFrame name with '_append' suffix
            new_df_name = f"{df_name}_append"

            # Check if the original DataFrame is present in dfs and not empty
            if df_name in dfs and not dfs[df_name].empty:
                # Find the row in material_df for the specified material
                material_row = material_df[
                    material_df["material_name"] == material_name
                ].drop("material_name", axis=1)
                # Replicate the material row to match the size of the building DataFrame
                repeated_material = pd.concat(
                    [material_row] * len(dfs[df_name]), ignore_index=True
                )
                # Create a new DataFrame by appending the material properties
                dfs[new_df_name] = pd.concat(
                    [
                        dfs[df_name].reset_index(drop=True),
                        repeated_material.reset_index(drop=True),
                    ],
                    axis=1,
                )
            # else:
            #     print(f"Skipped empty or missing DataFrame: {df_name}")

        return dfs

    def combine_dataframes_in_order(dfs, append_list):
        """
        Combines a list of DataFrames found in the dictionary 'dfs' into a single DataFrame,
        strictly following the order specified in 'append_list'. Skips any DataFrame names
        not found in the 'dfs' dictionary.

        Args:
        - dfs (dict): Dictionary containing DataFrames.
        - append_list (tuple): Tuple containing the names of the DataFrames to be combined in order.

        Returns:
        - pd.DataFrame: The combined DataFrame.
        """
        combined_df = pd.DataFrame()  # Initialize an empty DataFrame to start with

        for df_name in append_list:
            if df_name in dfs:  # Check if DataFrame name exists in the dictionary
                # If the combined DataFrame is empty, initialize it with the first DataFrame
                if combined_df.empty:
                    combined_df = dfs[df_name].copy()
                else:
                    # Concatenate the current DataFrame to the combined DataFrame
                    combined_df = pd.concat(
                        [combined_df, dfs[df_name]], ignore_index=True
                    )
            # else:
            #     print(
            #         f"DataFrame name '{df_name}' not found in the dictionary. Skipping..."
            #     )

        return combined_df

    def extract_values_to_dict(dfs, df_names):
        """
        Extracts a single value from each specified DataFrame and stores it in a dictionary.

        Args:
        - dfs (dict): Dictionary containing the DataFrames.
        - df_names (list of str): List of the names of the DataFrames to extract values from.

        Returns:
        - dict: A dictionary with the DataFrame names as keys and their extracted values as values.
        """
        values_dict = {}

        for df_name in df_names:
            if df_name in dfs and not dfs[df_name].empty:
                # Assuming each DataFrame contains only one value, extract it
                value = dfs[df_name].iloc[
                    0, 0
                ]  # Extract the first value of the DataFrame
                values_dict[df_name] = value
            # else:
            #     print(f"DataFrame '{df_name}' does not exist or is empty. Skipping...")

        return values_dict

    def map_sensor_to_vertex(sensor_length, values_dict):
        """
        Creates mappings of sensor IDs to vertex material IDs based on values_dict.

        Args:
        - sensor_length (int): The number of sensors.
        - values_dict (dict): Dictionary with vertex lengths for each material.

        Returns:
        - dict: A dictionary of DataFrames, each representing sensor to vertex mappings for a material.
        """
        mapped_dfs = {}
        for material, length in values_dict.items():
            material_name = material.split("_df")[
                0
            ]  # Extract material name from the key
            data = [
                (f"sensor_id_{sensor_id}", f"{material_name}_{i}")
                for sensor_id in range(1, sensor_length + 1)
                for i in range(1, length + 1)
            ]
            mapped_df = pd.DataFrame(data, columns=["sensor_id", "vertex_id"])
            mapped_dfs[material_name] = mapped_df

        return mapped_dfs

    def append_distance_to_mapped_dfs(mapped_dfs, dfs, material_list):
        """
        Correctly appends distance values from distance DataFrames in dfs to each corresponding mapped DataFrame.
        Correctly references the material name in mapped_dfs and skips materials if their corresponding distance DataFrame cannot be found in dfs.

        Args:
        - mapped_dfs (dict): Dictionary of mapped DataFrames.
        - dfs (dict): Dictionary containing distance DataFrames.
        - material_list (list): List of materials to search distance data for.

        Returns:
        - dict: Updated dictionary of mapped DataFrames with distance values appended.
        """
        for material in material_list:
            distance_df_name = f"distance_{material}_df"
            mapped_df_name = f"vertex_length_{material}"  # Adjusted to match the naming convention in mapped_dfs

            # Check if both the distance DataFrame and the mapped DataFrame exist
            if distance_df_name in dfs and mapped_df_name in mapped_dfs:
                distance_df = dfs[distance_df_name]
                # Assuming the distance values are in the first column of distance_df
                mapped_df = mapped_dfs[mapped_df_name]
                # Ensure the distance DataFrame has enough rows to match the mapped DataFrame
                if len(distance_df) >= len(mapped_df):
                    mapped_df["distance"] = distance_df.iloc[: len(mapped_df), 0].values
                else:
                    print(
                        f"Warning: Not enough distance values for {material}, distances not appended."
                    )
                mapped_dfs[mapped_df_name] = mapped_df
            # else:
            #     # Print a warning if the distance DataFrame is not found
            #     print(
            #         f"Warning: Distance DataFrame for '{material}' not found, skipping."
            #     )

        return mapped_dfs

    # Call function to import csv and convert to dataframe, storing them in a dictionary
    dfs = load_csv_files_as_dict(file_path_training, file_names)

    # Import Material Library and append to dfs dictionary of dataframes
    material_df = pd.read_csv(r"data\material\material_library.csv")
    dfs["material"] = material_df

    dfs = append_material_properties(dfs, material_df, df_material_map)

    # Create combined building_df
    building_df = combine_dataframes_in_order(dfs, append_list)

    values_dict = extract_values_to_dict(dfs, df_vertex_names)

    # extract out the length values and convert to numerical from both dataframes
    sensor_length = int(dfs["sensor_length_df"].iloc[0, 0])

    # Map sensor length with each material vertex length, return dictionary of dfs
    mapped_dfs = map_sensor_to_vertex(sensor_length, values_dict)

    final_dfs = append_distance_to_mapped_dfs(mapped_dfs, dfs, material_list)

    # Sort the keys of final_dfs alphabetically
    sorted_keys = sorted(final_dfs.keys())

    # Concatenate the DataFrames in alphabetical order
    edge_df = pd.concat([final_dfs[key] for key in sorted_keys], ignore_index=True)

    sensor_df = format_and_insert_id_column(dfs["sensor_df"], "sensor_id")
    building_df = format_and_insert_id_column(building_df, "vertex_id")
    label_df = format_and_insert_id_column(dfs["label_df"], "sensor_id")

    sensor_df.rename(
        columns={
            0: "sensor_x_coordinate",
            1: "sensor_y_coordinate",
            2: "sensor_z_coordinate",
        },
        inplace=True,
    )

    label_df.columns = ["sensor_id", "hb_solar_radiation"]

    # Add a column to distinguish between sensors and vertices
    sensor_df["type"] = "sensor"
    building_df["type"] = "vertex"

    # Combine dataframes
    all_nodes_df = pd.concat(
        [
            sensor_df.assign(index=range(0, len(sensor_df))),
            building_df.assign(
                index=range(len(sensor_df), len(sensor_df) + len(building_df))
            ),
        ]
    )

    # Prepare node features - example: using coordinates and a type flag (sensor=1, vertex=0)
    all_nodes_df["type_flag"] = all_nodes_df["type"].apply(
        lambda x: 1 if x == "sensor" else 0
    )
    node_features = (
        all_nodes_df[
            [
                "sensor_x_coordinate",
                "sensor_y_coordinate",
                "sensor_z_coordinate",
                "type_flag",
            ]
        ]
        .fillna(0)
        .values
    )
    x = torch.tensor(node_features, dtype=torch.float)

    sensor_ids = sensor_df["sensor_id"].unique()
    vertex_ids = building_df["vertex_id"].unique()

    # Create a continuous index for sensors and vertices
    sensor_index = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    vertex_index = {
        vertex_id: i + len(sensor_index) for i, vertex_id in enumerate(vertex_ids)
    }

    # Adjust the vertex_id in edge_df to match the format in buildings_df
    edge_df["adjusted_vertex_id"] = edge_df["vertex_id"].apply(
        lambda x: "vertex_id_" + x.split("_")[-1]
    )

    # Map sensor_id and vertex_id to their respective indices
    edge_index_list = edge_df.apply(
        lambda row: [
            sensor_index.get(row["sensor_id"], -1),
            vertex_index.get(row["adjusted_vertex_id"], -1),
        ],
        axis=1,
    )

    # Filter out any edges that couldn't be mapped (-1 indicates a mapping failure)
    filtered_edge_index_list = [pair for pair in edge_index_list if -1 not in pair]

    # Convert to torch tensor
    edge_index = (
        torch.tensor(filtered_edge_index_list, dtype=torch.long).t().contiguous()
    )

    edge_attr = torch.tensor(edge_df[["distance"]].values, dtype=torch.float)

    # Ensure data type compatibility
    label_df["hb_solar_radiation"] = label_df["hb_solar_radiation"].astype(float)

    # Update labels for sensors with their radiation values
    label_df["index"] = label_df["sensor_id"].map(sensor_index)

    # Create torch tensor with compatible data type
    labels = torch.zeros(len(label_df), dtype=torch.float)
    labels[label_df["index"]] = torch.tensor(
        label_df["hb_solar_radiation"].values, dtype=torch.float
    )

    placeholder_value = -1

    # Initialize mask tensors with zeros (False)
    train_mask = torch.zeros(len(sensor_ids), dtype=torch.bool)
    test_mask = torch.zeros(len(sensor_ids), dtype=torch.bool)

    # Example strategy: Use num_labeled_nodes for the current run to create masks
    labeled_indices = torch.arange(
        num_labeled_nodes
    )  # Assuming sequential labeling from 0

    # Random split for simplicity; consider a more sophisticated approach for real applications
    np.random.shuffle(labeled_indices.numpy())  # Convert to NumPy array for shuffling
    split_point = int(len(labeled_indices) * 0.8)

    train_indices = labeled_indices[:split_point]
    test_indices = labeled_indices[split_point:]

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    # labels[~train_mask & ~test_mask] = placeholder_value   #not needed as every node in label is used

    # Assuming 'train_mask' is a torch.Tensor
    train_indices = train_mask.nonzero(as_tuple=False).view(-1).tolist()

    # Append run number and training indices to the log
    training_mask_log.loc[len(training_mask_log)] = [run_num, train_indices]

    # Assuming train_mask and test_mask are your mask tensors
    num_train = train_mask.sum().item()
    num_test = test_mask.sum().item()
    total = len(train_mask)  # or test_mask, they are the same size

    print(f"Training nodes: {num_train} ({num_train/total*100:.2f}%)")
    print(f"Testing nodes: {num_test} ({num_test/total*100:.2f}%)")

    if pipeline_mode == "training":
        # Creating the Data object for training
        data_training = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        print("Data Object for Training:")
        print(data_training)
        torch.save(data_training, filename_training)
        # print(f"Training data saved as {filename_training}")

    elif pipeline_mode == "prediction":
        # Creating the Data object for prediction (without labels)
        data_predict = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # No y, train_mask, or test_mask needed for prediction
        )
        print("Data Object for Prediction:")
        print(data_predict)
        torch.save(data_predict, filename_prediction)
        # print(f"Prediction data saved as {filename_prediction}")


# Loop to process a range of runs, ensuring num_labeled_nodes is passed correctly
for i in range(1, 181):
    run_number = f"run_{i}"
    num_labeled_nodes = num_labeled_nodes_list[
        i - 1
    ]  # Assuming the list is in the same order as the runs
    print("-" * 50)
    print(f"Processing {run_number} with {num_labeled_nodes} labeled nodes...")
    process_run(run_number, num_labeled_nodes)
    print(f"Completed processing {run_number}.\n")
    print("-" * 50)

# Export the training mask log to a CSV file
training_mask_log.to_csv(
    r"data\torch_data_object_training\training_mask_log.csv", index=False
)

print("All runs processed.")
