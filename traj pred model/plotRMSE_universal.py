# 03/10/24:  Updated to change the colours of the RMSE plots
import torch
import time
import matplotlib.pyplot as plt # Addition
import itertools
from os.path import exists #The exists() function in Python exists in the os.path module, which is a submodule of
# the python’s OS module and is used to check if a particular file exists or not.

# Evaluation metric:
# scenes = ['US101_original' ,'StraightRd','HW Freeway']  #, 'Road Curve'] 
scenes = ['HW Freeway'] 
metrics = ['RMSE']  # or nll
vars = [''],#'Rain', 'Fog']
# Scenario settings:
sens = ['radar', 'camera']
features = ['FoV','Range']  
dirs = ['LLC','RLC' ]
# scenarios = ['Infront of Lead_data imputed','Infront of Ego_original','Infront of Ego_data imputed','Infront of Lead_original']
scenarios = ['Data imputed','Original']

# Unit mapping based on the feature        
unit = {'FoV': 'deg', 'Range': 'm'}

# Divisor mapping based on the feature  
divisor = {'FoV': 30, 'Range': 25}    

# Determine the rangeVals
rangeVals = {'FoV': range(30, 181, 30), 'Range': range(25, 151, 25)}

Spd = '5' # relative velocity

for scene, metric, var, sen, feature, dir, scenario in itertools.product(
    scenes, metrics, vars, sens, features, dirs, scenarios):

    # Determine the road abbreviation
    road_abbr = {
        'US101_original': 'US101',
        'HW Freeway': 'HW',
        'Road Curve': 'Curve',
        'StraightRd': 'StrRd'
    }  # [road] 

    path = scene + '/'+sen+'Data/manoeuvre_'+dir+'/'+feature+'/relVel_'+Spd+'kph/matFiles/'+scenario+'/'

    for m in rangeVals.get(feature,''):    
        file = path + road_abbr.get(scene,'') + '_' + feature + '_' + str(m) + '.pt'
        if exists(file):
            globals()[f"measured_{sen}_{dir}_{feature}_{m}"] = torch.load(file)
        # globals()[f"measured_{sen}_{dir}_{feature}_{m}"] = torch.load(path +Rd+'_'+ feature + var+'_'+str(m)+' occluded.pt')

    gTruth = torch.load(path+ road_abbr.get(scene,'')+'_groundTruth.pt')
    gTruth = gTruth.cpu()
    # gTruth = torch.load(path+ road_abbr+var+'_groundTruth occluded.pt')

    # Plotting the RMSE against prediction time
    plt.figure(1, figsize=(13, 5))
    title = (metric + ' Impact against changing '+sen +'_'+ feature + ' at relVel '+Spd+'kph_'+dir)
    plt.title(title, fontdict=None, loc='center')
    t = torch.arange(0, 5, 0.2)
    plt.plot(t, gTruth, color='blue', marker='o', label= metric +" on gTruth")

    # Set the colormap to be used for the curves
    cmap = plt.get_cmap('tab10')  # 'tab10' has 10 distinguishable colors

    # Counter for cycling through the colors in the colormap
    color_idx = 0

    for m in rangeVals.get(feature,''):
        file = path + road_abbr.get(scene,'') + '_' + feature + '_' + str(m) + '.pt'
        if exists(file):
            v = globals()[f"measured_{sen}_{dir}_{feature}_{m}"]
            v = v.cpu()
            # Cycle through colors using the colormap
            color = cmap(color_idx % cmap.N)  # Ensure it stays within the colormap's range
            color_idx += 1
            plt.plot(t, v, color=color, marker='o', label= metric+" on "+sen+" measured_" + feature +" "+str(m)+ unit.get(feature,''))
            # plt.plot(t, v, color=[m*5/1000, m*1.5/1000, m*3.5/1000], marker='o', label= metric+" on "+sen+" measured_" + feature +" "+str(m)+ unit)

    # plt.plot(t, gTruth, color='blue', marker='o', label= metric +" on gTruth")

    plt.grid(axis='y')
    plt.legend(loc="upper left")
    plt.xlabel('Time Frame (s)')
    plt.ylabel(metric+' (m)')
    plt.savefig(path+ metric+" Impact against changing "+sen+" "+feature+" @ "+Spd+"kph_"+road_abbr.get(scene,'')+".png")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

