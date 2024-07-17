import torch
import time
import matplotlib.pyplot as plt # Addition
from os.path import exists #The exists() function in Python exists in the os.path module, which is a submodule of
# the python’s OS module and is used to check if a particular file exists or not.

# Evaluation metric:
metric = 'RMSE'  # or 'NLL'
# Variable affecting sensor performance:
var = '' #Rain'  # or 'Fog, ''
# Road Geometry:
road = 'US101_original'  #'HW Freeway' #'Road Curve'#'US101_original'  #'StraightRd' #
scenario = 'Infront of Lead_original' #'Infront of Lead_data imputed' #'Infront of Ego_original' #'Infront of Ego_data imputed' #
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

# for m in range(30,181,30):
for m in range(50, 151, 25):
    file = path + Rd + '_' + feature + var + '_' + str(m) + '.pt'
    if exists(file):
        globals()[f"measured_{sen}_{dir}_{feature}_{m}"] = torch.load(file)
    # globals()[f"measured_{sen}_{dir}_{feature}_{m}"] = torch.load(path +Rd+'_'+ feature + var+'_'+str(m)+' occluded.pt')

gTruth = torch.load(path+ Rd+var+'_groundTruth.pt')
gTruth = gTruth.cpu()
# gTruth = torch.load(path+ Rd+var+'_groundTruth occluded.pt')

# Plotting the RMSE against prediction time
plt.figure(1, figsize=(13, 5))
title = (metric + ' Impact against changing '+sen +'_'+ feature + ' at relVel '+Spd+'kph_'+dir+'  '+var)
plt.title(title, fontdict=None, loc='center')
t = torch.arange(0, 5, 0.2)

plt.plot(t, gTruth, color='blue', marker='o', label= metric +" on gTruth")

# for m in range(30, 181, 30):
for m in range(50, 151, 25):
    file = path + Rd + '_' + feature + var + '_' + str(m) + '.pt'
    if exists(file):
        v = globals()[f"measured_{sen}_{dir}_{feature}_{m}"]
        v = v.cpu()
        plt.plot(t, v, color=[m*5/1000, m*1.5/1000, m*3.5/1000], marker='o', label= metric+" on "+sen+" measured_" + feature +" "+str(m)+ unit)

# plt.plot(t, gTruth, color='blue', marker='o', label= metric +" on gTruth")

plt.grid(axis='y')
plt.legend(loc="upper left")

plt.xlabel('Time Frame (s)')
plt.ylabel(metric+' (m)')
plt.savefig(path+ metric+" Impact against changing "+sen+" "+feature+" @ "+Spd+"kph_"+Rd+".png")
# plt.savefig(path+ metric+" Impact against changing "+sen+" "+feature+" @ "+Spd+"kph_"+Rd+"_Occluded.png")
plt.show()

