"""
This script is used to evaluate the performance of a model on a new dataset.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

model_name = "model_72"


# Placeholder function for rmse calculation, to be replaced with actual computation.
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


start_run = 1  # Starting run number
end_run = 9  # Ending run number (inclusive)
runs = [f"run_{i}" for i in range(start_run, end_run + 1)]

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["Run", "MAE", "RMSE", "R2"])

for run_num in runs:
    # Actual simulated labels
    label_actual_df = pd.read_csv(
        f"data/2_prediction_testing/actual_label_{run_num}.csv", header=None
    )
    # Predicted labels
    label_predict_df = pd.read_csv(
        f"data/2_prediction_testing/{model_name}_predictions_{run_num}.csv", header=None
    )

    actual_values = label_actual_df.iloc[:, 0]
    predicted_values = label_predict_df.iloc[:, 0]

    # Calculate MAE
    mae = mean_absolute_error(actual_values, predicted_values)

    # Calculate RMSE
    rmse_value = rmse(predicted_values, actual_values)

    # Calculate R2
    r2 = r2_score(actual_values, predicted_values)

    # Append results to DataFrame
    new_row = pd.DataFrame(
        {"Run": [run_num], "MAE": [mae], "RMSE": [rmse_value], "R2": [r2]}
    )
    results_df = pd.concat([results_df, new_row], ignore_index=True)


# Calculate averages of each metric
avg_metrics = results_df.mean(numeric_only=True)

# Creating a new row with these averages
# Note: 'Run' is set to 'Average' to distinguish this row
avg_row = pd.DataFrame(
    {
        "Run": ["Average"],
        "MAE": [avg_metrics["MAE"]],
        "RMSE": [avg_metrics["RMSE"]],
        "R2": [avg_metrics["R2"]],
    }
)

# Append the average row to the DataFrame
results_df = pd.concat([results_df, avg_row], ignore_index=True)


# Assuming the results_df to be correctly filled with metrics from the runs.
# Print the DataFrame to display results
print(results_df)

# Format the current timestamp into a string: YYYY-MM-DD_HH-MM-SS
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save the DataFrame, including the timestamp in the filename
results_df.to_csv(
    f"model_performance_metrics_{model_name}_{timestamp}.csv", index=False
)
