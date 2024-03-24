import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# import csv

run_num = "run_3"

# Actual simulated labels
label_actual_df = pd.read_csv(
    f"data/2_prediction_testing/actual_label_{run_num}.csv", header=None
)

# Predicted labels
label_predict_df = pd.read_csv(
    f"data/2_prediction_testing/sensor_predictions_{run_num}.csv", header=None
)
print("Files imported")

actual_values = label_actual_df.iloc[
    :, 0
]  # Selects all rows of the first (and only) column
predicted_values = label_predict_df.iloc[:, 0]  # Same as above

# Calculate MAE
mae = mean_absolute_error(actual_values, predicted_values)
print("Mean Absolute Error (MAE):", mae)


# Calculate RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


rmse = rmse(predicted_values, actual_values)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate R2
r2 = r2_score(actual_values, predicted_values)
print("R Squared Score (R2):", r2)
