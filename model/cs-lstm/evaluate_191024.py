# Added a 'seed' to set randon seeds and to configure PyTorch for deterministic behaviour

# Added line 53 to check if the the trajectory vector (traj) for the target vehicle is not empty in the relevant
# .mat file read using the 'ngsimDataset' dataset class and to use the updated 'utils_290923.py' version.

from __future__ import print_function
import torch
from model import highwayNet
from utils_041023 import ngsimDataset,maskedNLL,maskedMSETest,maskedNLLTest
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt # Addition
import numpy as np #Addition
import xlsxwriter # import xlsxwriter module
import itertools

def main():
    ## Network Arguments
    seed = 72
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = {}
    args['use_cuda'] = False
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 16
    args['out_length'] = 25
    args['grid_size'] = (13,3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 2
    args['use_maneuvers'] = False
    args['train_flag'] = False

        # Evaluation metric:
    metrics = ['rmse']  # or nll

    # Scenario settings:
    # scenes = ['US101_original' ,'StraightRd','HW Freeway']  #, 'Road Curve'] 
    scenes = ['HW Freeway'] 
    sens = ['radar', 'camera']
    params = ['Range', 'FoV']          
    dirs = ['RLC','LLC' ]
    # scenarios = ['Infront of Ego_original','Infront of Ego_data imputed','Infront of Lead_original', 'Infront of Lead_data imputed']
    scenarios = ['Data imputed','Original']

    # Unit mapping based on the feature        
    unit = {'FoV': 'deg', 'Range': 'm'}

    # Divisor mapping based on the feature  
    divisor = {'FoV': 30, 'Range': 25}     

    # Determine the rangeVals
    rangeVals = {'FoV': range(30, 181, 30), 'Range': range(25, 151, 25)}           

    # Determine the road abbreviation
    road_abbr = {
        'US101_original': 'US101',
        'HW Freeway': 'HW',
        'Road Curve': 'Curve',
        'StraightRd': 'StrRd'
    }  # [road] 

    # Determine the rangeVals
    rangeVals = {'FoV': range(30, 181, 30), 'Range': range(25, 151, 25)}

    # Initialize network
    net = highwayNet(args)
    net.load_state_dict(torch.load('trained_models/cslstm_no_m.tar'))
    # net.load_state_dict(torch.load('trained_models/cslstm_m.tar'))
    # net.load_state_dict(torch.load('trained_models/simData/cslstm_no_m_230523.tar', map_location=torch.device('cpu')))
    if args['use_cuda']:
        net = net.cuda()

    for metric, scene, param, sen, dir, scenario in itertools.product(
        metrics, scenes, params, sens, dirs, scenarios):
        
        print("RMSE calculations for the",sen+"'s",param,"for",dir,"on scenario:",scenario)
        
        rd = road_abbr.get(scene,'') 
        path = scene + '/'+sen+'Data/manoeuvre_'+dir+'/' + param + '/relVel_5kph/'
        f = open(path+ 'matFiles/'+scenario+'/'+'RMSE_'+rd+ '_'+dir+'.txt', 'w') # Open a text file to save the calculated RMSE values at each range setting
        book = xlsxwriter.Workbook(path+ 'matFiles/'+scenario+'/'+'RMSE_' +rd+ '_'+dir+'.xlsx')
        sheet = book.add_worksheet()
        cell_format = book.add_format({'bold': True, 'text_wrap': True, 'valign': 'top'})

        for a in rangeVals.get(param,''):
            tsSet = ngsimDataset(path + 'matFiles/'+scenario+'/' + scene +'_'+param+'_' + str(a) + '.mat')
            if (tsSet.D).size: #added to ensure that the trajectory vector for the target vehicle is not empty
                # tsSet = ngsimDataset(path + 'matFiles/Road Curve_Range_' + str(a) + '.mat')
                tsDataloader = DataLoader(tsSet, batch_size=1, shuffle=True, num_workers=1, collate_fn=tsSet.collate_fn, drop_last=True)

                lossVals = torch.zeros(25)#.cuda()
                counts = torch.zeros(25)#.cuda()

                for i, data in enumerate(tsDataloader):
                    st_time = time.time()
                    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
                    # Initialize Variables
                    if args['use_cuda']:
                        hist = hist.cuda()
                        nbrs = nbrs.cuda()
                        mask = mask.cuda()
                        lat_enc = lat_enc.cuda()
                        lon_enc = lon_enc.cuda()
                        fut = fut.cuda()
                        op_mask = op_mask.cuda()

                    if metric == 'nll':
                        # Forward pass
                        if args['use_maneuvers']:
                            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                            l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
                        else:
                            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                            l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask, use_maneuvers=False)
                    else:
                        # Forward pass
                        if args['use_maneuvers']:
                            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                            fut_pred_max = torch.zeros_like(fut_pred[0])
                            for k in range(lat_pred.shape[0]):
                                lat_man = torch.argmax(lat_pred[k, :]).detach()
                                lon_man = torch.argmax(lon_pred[k, :]).detach()
                                indx = lon_man*3 + lat_man
                                fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
                            l, c = maskedMSETest(fut_pred_max, fut, op_mask)
                        else:
                            fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                            l, c = maskedMSETest(fut_pred, fut, op_mask)

                    lossVals +=l.detach()
                    counts += c.detach()

                # rmse = (torch.pow(lossVals / counts, 0.5) * 0.3048)
                if metric == 'nll':
                    nll = (lossVals / counts)
                    loss = nll
                else:
                    rmse = (torch.pow(lossVals / counts, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
                    loss = rmse

                #print('something', file=f)
                print('RMSE at '+param+' ' + str(a), loss)
                print('RMSE at '+param+' ' + str(a), loss, file=f)  # Calculate RMSE and convert from feet to meters

                # Rows and columns are zero indexed.
                row = 1
                column = int(a/divisor.get(param,''))
                sheet.write(0, column, 'RMSE(m) at '+param+' '+str(a)+ unit.get(param,''), cell_format)
                # iterating through the content list
                for item in rmse:
                    if torch.isnan(item):
                        item = 'nan'
                    else:
                        item = round(item.numpy().item(),2)
                    # write operation perform
                    sheet.write(row, column, item)
                    # incrementing the value of row by one with each iterations.
                    row += 1

                torch.save(rmse, path +'matFiles/'+scenario+'/'+ rd+ '_'+param+'_' + str(a) + '.pt')
        # book.close()

        gtSet = ngsimDataset(path + 'matFiles/'+scenario+'/'+ scene + '_groundTruth.mat')
        gtDataloader = DataLoader(gtSet, batch_size=1, shuffle=True, num_workers=1, collate_fn=gtSet.collate_fn, drop_last=True)  # Added parameter, 'drop_last' and changed 'num_workers' to 1

        lossVals = torch.zeros(25)  # .cuda() # should .cuda() be removed?
        counts = torch.zeros(25)  # .cuda()

        for i, data in enumerate(gtDataloader):
            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            # Initialize Variables
            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()

            if metric == 'nll':
                # Forward pass
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)
                else:
                    fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask, use_maneuvers=False)
            else:
                # Forward pass
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    fut_pred_max = torch.zeros_like(fut_pred[0])
                    for k in range(lat_pred.shape[0]):
                        lat_man = torch.argmax(lat_pred[k, :]).detach()
                        lon_man = torch.argmax(lon_pred[k, :]).detach()
                        indx = lon_man * 3 + lat_man
                        fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]
                    l, c = maskedMSETest(fut_pred_max, fut, op_mask)
                else:
                    fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedMSETest(fut_pred, fut, op_mask)

            lossVals += l.detach()
            counts += c.detach()

        err = (torch.pow(lossVals / counts, 0.5) * 0.3048)
        # if metric == 'nll':
        #     print(lossVals / counts)
        # else:
        #     err = (torch.pow(lossVals / counts, 0.5) * 0.3048)

        print('RMSE on gndTruth: ', err)
        print('RMSE on gndTruth ', err, file=f)  # Calculate RMSE and convert from feet to meters

        row = 1
        sheet.write(0, 0, 'GroundTruth RMSE(m)',cell_format)
        for item in err:
            if torch.isnan(item):
                item = 'nan'
            else:
                item = round(item.numpy().item(), 2)
            # for item in err.numpy():
            # write operation perform
            sheet.write(row, 0, item)
            # incrementing the value of row by one with each iterations.
            row += 1

        torch.save(err, path +'matFiles/'+scenario+'/'+ rd +'_groundTruth.pt')

        f.close() # Close the text file
        # Color-coding the minimum RMSE value at each timeframe in the prediction horizon
        format1 = book.add_format({'bg_color': '#B7DBFF'})
        for rNum in range(26):
            sheet.conditional_format('C{i}:G{j}'.format(i=rNum+1,j=rNum+1), {"type": "bottom", "value": 1, "format": format1})
        book.close()

if __name__ == "__main__":
    main()
