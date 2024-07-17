import torch
import pandas as pd
import numpy as np
import os

def calculate_sprague_geers(observed, predicted):
    n = len(observed)
    bias_error = (1 / n) * sum((predicted[i] - observed[i]) / observed[i] for i in range(n)) * 100
    phase_error = (1 / n) * sum(abs(predicted[i] - observed[i]) / observed[i] for i in range(n)) * 100
    combined_error = (bias_error**2 + phase_error**2)**0.5

    pearson_correlation = np.corrcoef(observed, predicted)[0,1]

    return round(bias_error, 2), round(phase_error, 2), round(combined_error, 2), round(pearson_correlation,2)

def load_rmse_values(file_path):
    data = torch.load(file_path)
    rmse_values = data#['rmse_values']
    if isinstance(rmse_values, torch.Tensor):
        rmse_values = rmse_values.tolist()
    # print(f"Loaded RMSE values from {file_path}: {rmse_values}")
    return rmse_values

# Evaluation metric:
metric = 'RMSE'  # or 'NLL'
# Variable affecting sensor performance:
var = '' #Rain'  # or 'Fog, ''
# Road Geometry:
road = 'US101_original' #'HW Freeway' #'Road Curve'#'US101_original'  #'StraightRd' #
scenario = 'Infront of Lead_data imputed' #'Infront of Ego_original' #'Infront of Ego_data imputed' # 'Infront of Lead_original' #
# Lane change direction:
dir = 'RLC' #'LLC' #
# Sensor:
sen = 'camera' #'radar'#
feature = 'Range'#'FoV' #
Spd = '5' # relative velocity

if road == 'HW Freeway':
    Rd = 'HW'
elif road == 'US101_original':
    Rd = 'US101'
elif road == 'Road Curve':
    Rd = 'Curve'
elif road == 'StraightRd':
    Rd = 'StrRd'

if feature == 'Range':
    unit = 'm'
elif feature == 'FoV':
    unit = 'deg'

path = road + '/'+sen+'Data/manoeuvre_'+dir+'/'+feature+'/relVel_'+Spd+'kph/matFiles/'+scenario+'/'

def main():
    # Folder containing your .pt files
    folder_path = path
    
    # Ground truth RMSE file
    ground_truth_file = path + Rd+var+'_groundTruth.pt'
    
    # Load ground truth RMSE values
    ground_truth_rmse = load_rmse_values(ground_truth_file)
    
    # Prepare a list to hold results
    results = []
    
    # Loop through each .pt file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pt') and file_name != os.path.basename(ground_truth_file):
            file_path = os.path.join(folder_path, file_name)
            sensor_rmse = load_rmse_values(file_path)
            
            # Calculate errors
            bias_error, phase_error, combined_error, pearson_correlation = calculate_sprague_geers(ground_truth_rmse, sensor_rmse)
            
            # Extract numerical value from the file name (assuming file names contain the sensor parameter values)
            try:
                sensor_param = int(''.join(filter(str.isdigit, file_name)))
            except ValueError:
                sensor_param = None
            
            # Add results to the list
            results.append({
                'Sensor Param': sensor_param,
                'Bias Error (%)': bias_error,
                'Phase Error (%)': phase_error,
                'Combined Error (%)': combined_error,
                'Pearson Correlation': pearson_correlation
            })
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    # df = pd.DataFrame(results, index=['Bias Error (%)', 'Phase Error (%)', 'Combined Error (%)', 'Pearson Correlation']).reset_index() #NEW
    
    # Sort DataFrame by 'Sensor Param'
    df.sort_values(by='Sensor Param', inplace=True)
    
    # Set 'Sensor Param' as columns and error types as rows
    df_pivot = df.pivot_table(index=None, columns='Sensor Param', values=['Bias Error (%)', 'Phase Error (%)', 'Combined Error (%)', 'Pearson Correlation'])#.transpose()
    
    # Save transposed DataFrame to Excel
    output_file = path+'Sprague_Geers_errors_'+scenario+'.xlsx'
    df_pivot.to_excel(output_file, index=True)
    print(f'Results saved to {output_file}')

if __name__ == '__main__':
    main()