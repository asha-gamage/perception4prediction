import torch
import pandas as pd
import numpy as np
import os
import itertools
from scipy.stats import pearsonr, spearmanr
from scipy.integrate import simps
import xlsxwriter 

# Function to extract the feature value (FoV or Range) from the filename
def extract_feature_value(filename, feature, road_abbr, road):
    # Use the appropriate road abbreviation based on the value of 'road'
    road_prefix = road_abbr.get(road, '')
    # Construct the prefix based on the road abbreviation and the feature
    prefix = f"{road_prefix}_{feature}_"
    if prefix in filename:
        start_idx = filename.find(prefix) + len(prefix)
        end_idx = filename.find('.pt', start_idx)
        return filename[start_idx:end_idx]
    
    return 'Unknown'

# Function to compute Signal to Noise Ratio (SNR)
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

# Function to calculate correlation coefficients
def calculate_coefficients(ground_truth_tensor, comparison_tensor):
    pearson_coeff, _ = pearsonr(ground_truth_tensor, comparison_tensor)
    spearman_coeff, _ = spearmanr(ground_truth_tensor, comparison_tensor)
    return pearson_coeff, spearman_coeff

# Function to create a results table and save it to Excel with formatting
def save_results_to_excel(results, feature, output_path, scenario, unit):
    # Sort the feature values in ascending order
    sorted_results = {f"{key}{unit}": results[key] for key in sorted(results.keys(), key=lambda x: float(x))}
    
    # Create a DataFrame from the results dictionary
    df = pd.DataFrame(sorted_results, index=['Pearson Coeff', 'Spearman Coeff', 'Max Error','MAE', 'AUC', 'SNR'])

    # Set the columns to be the sorted feature values and index as the metrics
    df.columns.name = f'{feature} Value({unit})'
    df.index.name = 'Metric'

    # Define the output Excel file path
    output_file = os.path.join(output_path, f"{scenario}_{feature}_correlation_coefficients.xlsx")
    
    # Use XlsxWriter as the engine for advanced formatting
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # Define baby blue format
        baby_blue = workbook.add_format({'bg_color': '#ADD8E6'})

        # Iterate through each metric to find and highlight the best value(s)
        for row_num, metric in enumerate(df.index):
            if metric in ['Pearson Coeff', 'Spearman Coeff']:
                # For Pearson and Spearman, find the column(s) closest to 1
                closest_distance = df.loc[metric].apply(lambda x: abs(x - 1)).min()
                best_columns = df.columns[df.loc[metric].apply(lambda x: abs(x - 1) == closest_distance)]
            elif metric in ['MAE', 'AUC', 'Max Error']:
                # For MAE, AUC, Max Error, find the minimum value
                min_value = df.loc[metric].min()
                best_columns = df.columns[df.loc[metric] == min_value]
            elif metric == 'SNR':
                # For SNR, find the maximum value
                max_value = df.loc[metric].max()
                best_columns = df.columns[df.loc[metric] == max_value]
            else:
                continue  # Skip unknown metrics

            # Apply the baby blue format to the best cell(s) in the current row
            for col_num, column in enumerate(df.columns):
                if column in best_columns:
                    # Calculate the Excel cell location (row_num +1 for header, col_num +1 for index)
                    cell_row = row_num + 1  # Excel rows start at 1 (0 is header)
                    cell_col = col_num + 1  # Excel columns start at 1 (0 is index)
                    cell_location = xlsxwriter.utility.xl_rowcol_to_cell(cell_row, cell_col)
                    worksheet.write(cell_row, cell_col, df.loc[metric, column], baby_blue)

        print(f"Results saved to: {output_file}")

# Function to identify and load the ground truth file in a given directory
def load_ground_truth_file(pt_directory):
    for pt_file in os.listdir(pt_directory):
        if 'groundTruth' in pt_file and pt_file.endswith('.pt'):
            ground_truth_path = os.path.join(pt_directory, pt_file)
            return torch.load(ground_truth_path).numpy()
    return None

# Variables for all possible combinations
metrics = ['RMSE']  # Extend this list if needed
vars_ = ['', 'Rain', 'Fog']  # Add more variations as required
# roads = ['US101_original', 'StraightRd','HW Freeway', 'Road Curve']  # Extend as needed
roads = ['HW Freeway']  # Extend as needed
scenarios = ['Infront of Lead_data imputed','Infront of Ego_original','Infront of Ego_data imputed','Infront of Lead_original','Data imputed','Original']
directions = ['LLC', 'RLC']
sensors = ['camera', 'radar']
features = ['FoV', 'Range']
speeds = ['5']  # Extend if there are more speed variations

# Unit mapping based on the feature
units = {'FoV': 'deg', 'Range': 'm'}

# Iterate through all combinations of variables
for metric, var, road, scenario, direction, sensor, feature, speed in itertools.product(
    metrics, vars_, roads, scenarios, directions, sensors, features, speeds):
    
    # Determine the road abbreviation
    road_abbr = {
        'US101_original': 'US101',
        'HW Freeway': 'HW',
        'Road Curve': 'Curve',
        'StraightRd': 'StrRd'
    }  # [road]

    # Define the directory path to the .pt files based on the current combination
    pt_directory = f"{road}/{sensor}Data/manoeuvre_{direction}/{feature}/relVel_{speed}kph/matFiles/{scenario}/"

    # Directory where output Excel files will be saved
    output_base_dir = f"{road}/{sensor}Data/manoeuvre_{direction}/{feature}/relVel_{speed}kph/matFiles/"  

    # Check if the directory exists
    if not os.path.exists(pt_directory):
        print(f"Directory not found: {pt_directory}")
        continue

    # Load the ground truth tensor from the directory
    ground_truth_tensor = load_ground_truth_file(pt_directory)
    ground_truth_np = ground_truth_tensor
    if ground_truth_tensor is None:
        print(f"Ground truth file not found in directory: {pt_directory}")
        continue

    # Dictionary to hold correlation results for this combination
    results = {}

    # Iterate through all .pt files in the directory
    for pt_file in os.listdir(pt_directory):
        # Skip the ground truth file
        if 'groundTruth' in pt_file:
            continue

        if pt_file.endswith('.pt'):
            # Extract the feature value from the filename
            feature_value = extract_feature_value(pt_file, feature, road_abbr, road)

            # Load the comparison tensor
            comparison_tensor = torch.load(os.path.join(pt_directory, pt_file)).numpy()

            # Ensure ground truth and predicted RMSEs are numpy arrays
            predicted_rmse_np = comparison_tensor

            # Calculate the correlation coefficients
            pearson_coeff, spearman_coeff = calculate_coefficients(ground_truth_tensor, comparison_tensor)
                        # Max Error
            max_error = np.max(np.abs(ground_truth_np - predicted_rmse_np))

            # Mean Absolute Error (MAE)
            mae = np.mean(np.abs(ground_truth_np - predicted_rmse_np))
    
            # Area Under the Curve (AUC) for RMSE vs Time
            auc = simps(predicted_rmse_np)  # Integrating RMSE values over the prediction horizon (time)

             # Signal to Noise Ratio (SNR)
            snr = calculate_snr(ground_truth_np, ground_truth_np - predicted_rmse_np)

            # Store the coefficients in the results dictionary
            results[feature_value] = [
                round(pearson_coeff, 4),
                round(spearman_coeff, 4),
                round(max_error, 4),
                round(mae, 4),
                round(auc, 4),
                round(snr, 4)
                ]

    # Get the unit for the current feature
    unit = units[feature]

    # Create the output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Save the results to an Excel file with formatting
    save_results_to_excel(results, feature, output_base_dir, scenario, unit)