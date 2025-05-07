# 03/10/24:  Updated to change the colours of the RMSE plots
import torch
import time
import matplotlib.pyplot as plt # Addition
import itertools
from os.path import exists #The exists() function in Python exists in the os.path module, which is a submodule of
# the python’s OS module and is used to check if a particular file exists or not.

# Evaluation metric:
# scenes = ['US101_original' ,'StraightRd','HW Freeway']  #, 'Road Curve'] 
scenes = ['US101_original'] 
metrics = ['RMSE']  # or nll
vars = [''],#'Rain', 'Fog']
# Scenario settings:
# sens = ['radar', 'camera']
sens = ['camera']
features = ['Range'] #['Range', 'FoV', 'Angular Res', 'Range Res', 'Resolution']   
dirs = ['RLC','LLC'] #, 'Decel']
scenarios = ['Infront of Ego_original','Infront of Lead_original']
# scenarios = ['Infront of Lead_data imputed','Infront of Ego_original','Infront of Ego_data imputed','Infront of Lead_original']#, 'Deceleration_original', 'Deceleration_data imputed']
# scenarios = ['Data imputed','Original']
# scenarios = ['Deceleration_original', 'Deceleration_data imputed']

# Unit mapping based on the feature        
unit = {'FoV': 'deg', 'Range': 'm', 'Angular Res':'deg', 'Range Res': 'cm', 'Resolution': 'px'}

# Divisor mapping based on the feature  
divisor = {'FoV': 30, 'Range': 25, 'Range Res': 20, 'Angular Res': 2, 'Resolution': 1}    

# Determine the rangeVals
rangeVals = {'FoV': range(30, 181, 30), 'Range': range(50, 151, 25), 'Range Res': range (20, 101, 20), 'Angular Res': range (2, 11, 2), 'Resolution': ['1280x720','1920x1080','2560x1440','3840x2160','7680x4320']}

Spd = '5' # relative velocity

for scene, metric, var, sen, feature, dir, scenario in itertools.product(
    scenes, metrics, vars, sens, features, dirs, scenarios):
    
    if sen == 'camera' and feature in ['Angular Res', 'Range Res']:
        continue  # Skip this iteration if the feature is not valid for 'camera'
    if sen == 'radar' and feature == 'Resolution':
        continue
    if dir == 'Decel' and scenario in ['Infront of Ego_original','Infront of Ego_data imputed','Infront of Lead_original', 'Infront of Lead_data imputed']:
        continue
    if (dir == 'RLC' or dir == 'LLC') and scenario in ['Deceleration_original', 'Deceleration_data imputed']:
        continue      

    # Determine the road abbreviation
    road_abbr = {
        'US101_original': 'US101',
        'HW Freeway': 'HW',
        'Road Curve': 'Curve',
        'StraightRd': 'StrRd'
    }  
    if dir == "Decel" :
        path = scene + '/'+sen+'Data/manoeuvre_'+dir+'/' + feature + '/matFiles/'+scenario+'/'
    else:
        path = scene + '/'+sen+'Data/manoeuvre_'+dir+'/' + feature + '/relVel_'+Spd+'kph/matFiles/'+scenario+'/'

    for m in rangeVals.get(feature,''): 
        if feature == 'Resolution':
            file = path + road_abbr.get(scene,'') + '_' + feature + '_' + m + '.pt' 
        else:  
            file = path + road_abbr.get(scene,'') + '_' + feature + '_' + str(m) + '.pt'

        if exists(file):
            globals()[f"measured_{sen}_{dir}_{feature}_{m}"] = torch.load(file)
        # globals()[f"measured_{sen}_{dir}_{feature}_{m}"] = torch.load(path +Rd+'_'+ feature + var+'_'+str(m)+' occluded.pt')

    gTruth = torch.load(path+ road_abbr.get(scene,'')+'_groundTruth.pt')
    gTruth = gTruth.cpu()
        # gTruth = torch.load(path+ road_abbr+var+'_groundTruth occluded.pt')

    # Plotting the RMSE against prediction time
    plt.figure(1, figsize=(11, 5))
    title = (metric + ' Impact against changing '+sen +'_'+ feature + ' at relVel '+Spd+'kph_'+dir)
    plt.title(title, fontdict=None, loc='center', fontsize=20)
    t = torch.arange(0, 5, 0.2)
    #plt.plot(t, gTruth, color='blue', marker='o', label= metric +" on gTruth")
    plt.plot(t, gTruth, color='blue', marker='o', label= "gTruth")

    # Set the colormap to be used for the curves
    cmap = plt.get_cmap('tab10')  # 'tab10' has 10 distinguishable colors

    # Counter for cycling through the colors in the colormap
    color_idx = 0

    for m in rangeVals.get(feature,''):
        if feature == 'Resolution':
            file = path + road_abbr.get(scene,'') + '_' + feature + '_' + m + '.pt'
        else:
            file = path + road_abbr.get(scene,'') + '_' + feature + '_' + str(m) + '.pt'
        if exists(file):
            v = globals()[f"measured_{sen}_{dir}_{feature}_{m}"]
            v = v.cpu()
            #Cycle through colors using the colormap
            color = cmap(color_idx % cmap.N)  # Ensure it stays within the colormap's range
            color_idx += 1
            if feature == 'Resolution':        
                plt.plot(t, v, color=color, marker='o', label=f"{feature} {m}"+ unit.get(feature,''))
            else:
                plt.plot(t, v, color=color, marker='o', label= feature +" "+str(m)+ unit.get(feature,''))
            # plt.plot(t, v, color=[m*5/1000, m*1.5/1000, m*3.5/1000], marker='o', label= metric+" on "+sen+" measured_" + feature +" "+str(m)+ unit)
    plt.grid(axis='y')
    plt.ylim(bottom=0, top=5.5)  # Ensure y-axis starts from zero
    plt.xlim(left=-0.025)
    plt.legend(loc="upper left", fontsize=16.5)
    plt.xlabel('Time Frame (s)', fontsize=18)
    plt.ylabel(metric+' (m)', fontsize=18)
    # Set the tick markers font size
    plt.tick_params(axis='both', labelsize=16.5)  # Increased font size for tick labels
    plt.tight_layout()
    #plt.savefig(path+ metric+" Impact against changing "+sen+" "+feature+" @ "+Spd+"kph_"+road_abbr.get(scene,'')+".png")
    plt.savefig(path+" Impact against changing "+sen+" "+feature+" @ "+Spd+"kph_"+road_abbr.get(scene,'')+".png")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

