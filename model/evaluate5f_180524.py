from __future__ import print_function

import loader2_150424 as lo
from torch.utils.data import DataLoader
import pandas as pd
from config import *
import matplotlib.pyplot as plt
import os
import time
import torch # added 12/12/23
import xlsxwriter # import xlsxwriter module 18/12/23

class Evaluate():

    def maskedMSETest(self, y_pred, y_gt, mask):
        y_gt = y_gt.squeeze()  # Remove the dimension for the sampleID
        mask = mask.squeeze()
        y_pred = y_pred.squeeze()
        y_gt_avbl = y_gt[y_gt.sum(dim=1) != 0]
        n = y_gt_avbl.shape[0]
        if n < 25:
            y_pred = torch.cat((y_pred[0:n, :], torch.zeros_like(y_pred[n:y_pred.shape[0], :])), dim=0)

        acc = torch.zeros_like(mask)
        muX = y_pred[:, 0]
        muY = y_pred[:, 1]
        x = y_gt[:, 0]
        y = y_gt[:, 1]
        out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
        lossVal = out
        counts = mask[:, 0]
        return lossVal, counts

    ## Helper function for log sum exp calculation: 一个计算公式
    def logsumexp(self, inputs, dim=None, keepdim=False):
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = t.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=2,
                      use_maneuvers=True):
        if use_maneuvers:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
            count = 0
            for k in range(num_lon_classes):
                for l in range(num_lat_classes):
                    wts = lat_pred[:, l] * lon_pred[:, k]
                    wts = wts.repeat(len(fut_pred[0]), 1)
                    y_pred = fut_pred[k * num_lat_classes + l]
                    y_gt = fut
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = t.pow(1 - t.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    # If we represent likelihood in feet^(-1):
                    out = -(0.5 * t.pow(ohr, 2) * (
                            t.pow(sigX, 2) * t.pow(x - muX, 2) + 0.5 * t.pow(sigY, 2) * t.pow(
                        y - muY, 2) - rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                                    y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    acc[:, :, count] = out + t.log(wts)
                    count += 1
            acc = -self.logsumexp(acc, dim=2)
            acc = acc * op_mask[:, :, 0]
            loss = t.sum(acc) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc, dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
        else:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
            y_pred = fut_pred
            y_gt = fut
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # p
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            # If we represent likelihood in feet^(-1):
            out = 0.5 * t.pow(ohr, 2) * (
                    t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY,
                                                                                2) - 2 * rho * t.pow(
                sigX, 1) * t.pow(sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
            acc[:, :, 0] = out
            acc = acc * op_mask[:, :, 0:1]
            loss = t.sum(acc[:, :, 0]) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc[:, :, 0], dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss

    def main(self, name, val):

        # Evaluation metric:
        metric = 'rmse'  # or nll

        # Scenario settings:
        sen = 'radar' #'camera'  #
        param = 'Range'  #'FoV' #
        scene = 'US101_original' #'StraightRd' #'HW Freeway'  #or 'Road Curve' #  #
        dir = 'RLC'  #'LLC' #
        scenario = 'Infront of Lead_original' #'Infront of Lead_data imputed' #'Infront of Ego_original' #'Infront of Ego_data imputed' #'Infront of Lead_original' #

        if param == 'Range':
            unit = 'm'
            divisor = 25
        elif param == 'FoV':
            unit = 'deg'
            divisor = 30

        if scene == 'HW Freeway':
            rd = 'HW'
        elif scene == 'US101_original':
            rd = 'US101'
        elif scene == 'Road Curve':
            rd = 'Curve'
        elif scene == 'StraightRd':
            rd = 'StrRd'

        model_step = 1
        # args['train_flag'] = not args['use_maneuvers']
        args['train_flag'] = True  #???
        l_path = args['path']
        generator = model.Generator(args=args)
        gdEncoder = model.GDEncoder(args=args)
        generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.tar', map_location='cpu')) # changed from 'cuda:1' to 'cuda:0': 02/12/23
        gdEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_gd.tar', map_location='cpu'))# changed from 'cuda:1' to 'cuda:0': 02/12/23
        generator = generator.to(device)
        gdEncoder = gdEncoder.to(device)
        generator.eval()
        gdEncoder.eval()

        path = scene + '/' + sen + 'Data/manoeuvre_' + dir + '/' + param + '/relVel_5kph/'
        f = open(path + 'matFiles/'+scenario+'/RMSE_' + rd + '_' + dir + '.txt',
                 'w')  # Open a text file to save the calculated RMSE values at each range setting
        book = xlsxwriter.Workbook(path +'matFiles/'+scenario+ 'RMSE_' + rd + '_' + dir + '.xlsx')
        sheet = book.add_worksheet()
        cell_format = book.add_format({'bold': True, 'text_wrap': True, 'valign': 'top'})

        for a in range(25, 151, 25):
        # for a in range(30, 181, 30):
            t2 = lo.NgsimDataset(path + 'matFiles/'+scenario+'/' + scene +'_'+param+'_' + str(a) + '.mat')
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_worker'],
                               collate_fn=t2.collate_fn) 

            lossVals = t.zeros(args['out_length']).to(device)
            counts = t.zeros(args['out_length']).to(device)
            avg_val_loss = 0
            all_time = 0
            nbrsss = 0

            val_batch_count = len(valDataloader)
            print("RMSE at............", param,':',a,unit)
            with(t.no_grad()):
                for idx, data in enumerate(valDataloader):
                    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data
                    hist = hist.to(device)
                    nbrs = nbrs.to(device)
                    mask = mask.to(device)
                    lat_enc = lat_enc.to(device)
                    lon_enc = lon_enc.to(device)
                    fut = fut[:args['out_length'], :, :]
                    fut = fut.to(device)
                    op_mask = op_mask[:args['out_length'], :, :]
                    op_mask = op_mask.to(device)
                    va = va.to(device)
                    nbrsva = nbrsva.to(device)
                    lane = lane.to(device)
                    nbrslane = nbrslane.to(device)
                    cls = cls.to(device)
                    nbrscls = nbrscls.to(device)
                    map_positions = map_positions.to(device)
                    te = time.time()
                    values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
                    fut_pred, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
                    all_time += time.time() - te

                    if not args['train_flag']:
                        indices = []
                        if args['val_use_mse']:
                            fut_pred_max = t.zeros_like(fut_pred[0])
                            for k in range(lat_pred.shape[0]):  # 128
                                lat_man = t.argmax(lat_enc[k, :]).detach()
                                lon_man = t.argmax(lon_enc[k, :]).detach()
                                index = lon_man * 3 + lat_man
                                indices.append(index)
                                fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
                            l, c, loss = self.maskedMSETest(fut_pred_max, fut, op_mask)
                        else:
                            l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                            use_maneuvers=args['use_maneuvers'])
                    else:
                        if args['val_use_mse']:
                            l, c = self.maskedMSETest(fut_pred, fut, op_mask)
                        else:
                            l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                            use_maneuvers=args['use_maneuvers'])

                    lossVals += l.detach()
                    counts += c.detach()
                    #avg_val_loss += loss.item()
                    if idx == int(val_batch_count / 4) * model_step:
                        print('process:', model_step / 4)
                        model_step += 1

                if args['val_use_mse']:
                    print('valmse:', avg_val_loss / val_batch_count)
                    rmse = (t.pow(lossVals / counts, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
                    # rmse = (t.pow(lossVals / counts, 0.5) )  # Calculate RMSE and convert from feet to meters
                    print(rmse)
                else:
                    print('valnll:', avg_val_loss / val_batch_count)
                    print(lossVals / counts)
                # Rows and columns are zero indexed.

            row = 1
            column = int(a / divisor)
            sheet.write(0, column, 'RMSE(m) at ' + param + ' ' + str(a) + unit, cell_format)
            # iterating through the content list
            for item in rmse:
                item=item.cpu()
                if torch.isnan(item):
                    item = 'nan'
                else:
                    item = round(item.numpy().item(), 2)
                    # write operation perform
                sheet.write(row, column, item)
                # incrementing the value of row by one with each iterations.
                row += 1

            torch.save(rmse, path +'matFiles/'+scenario+'/'+ rd + '_' + param + '_' + str(a) + '.pt')

     
        t2 = lo.NgsimDataset(path + 'matFiles/'+scenario+'/' + scene + '_groundTruth.mat')
        valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_worker'],
                                   collate_fn=t2.collate_fn) 

        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        all_time = 0
        nbrsss = 0

        val_batch_count = len(valDataloader)
        print("RMSE at............groundTruth")
        with(t.no_grad()):
            for idx, data in enumerate(valDataloader):
                hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data
                hist = hist.to(device)
                nbrs = nbrs.to(device)
                mask = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)
                fut = fut[:args['out_length'], :, :]
                fut = fut.to(device)
                op_mask = op_mask[:args['out_length'], :, :]
                op_mask = op_mask.to(device)
                va = va.to(device)
                nbrsva = nbrsva.to(device)
                lane = lane.to(device)
                nbrslane = nbrslane.to(device)
                cls = cls.to(device)
                nbrscls = nbrscls.to(device)
                map_positions = map_positions.to(device)
                te = time.time()
                values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
                fut_pred, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)
                all_time += time.time() - te

                if not args['train_flag']:
                    indices = []
                    if args['val_use_mse']:
                        fut_pred_max = t.zeros_like(fut_pred[0])
                        for k in range(lat_pred.shape[0]):  # 128
                            lat_man = t.argmax(lat_enc[k, :]).detach()
                            lon_man = t.argmax(lon_enc[k, :]).detach()
                            index = lon_man * 3 + lat_man
                            indices.append(index)
                            fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
                        l, c, loss = self.maskedMSETest(fut_pred_max, fut, op_mask)
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                    if self.drawImg:
                        lat_man = t.argmax(lat_enc, dim=-1).detach()
                        lon_man = t.argmax(lon_enc, dim=-1).detach()
                        self.draw(hist, fut, nbrs, mask, fut_pred, args['train_flag'], lon_man, lat_man, op_mask, indices)
                else:
                    if args['val_use_mse']:
                        l, c = self.maskedMSETest(fut_pred, fut, op_mask)
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                lossVals += l.detach()
                counts += c.detach()
                # avg_val_loss += loss.item()
                if idx == int(val_batch_count / 4) * model_step:
                    print('process:', model_step / 4)
                    model_step += 1

            if args['val_use_mse']:
                print('valmse:', avg_val_loss / val_batch_count)
                err = (t.pow(lossVals / counts, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
                # err = (t.pow(lossVals / counts, 0.5))  # Calculate RMSE and convert from feet to meters
                print(err)
            else:
                print('valnll:', avg_val_loss / val_batch_count)
                print(lossVals / counts)
            # Rows and columns are zero indexed.

        row = 1
        sheet.write(0, 0, 'GroundTruth RMSE(m)',cell_format)
        # iterating through the content list
        for item in err:
            item = item.cpu()
            if torch.isnan(item):
                item = 'nan'
            else:
                item = round(item.numpy().item(), 2)
                # write operation perform
            sheet.write(row, 0, item)
            # incrementing the value of row by one with each iterations.
            row += 1
        torch.save(err, path +'matFiles/'+scenario+'/'+ rd + '_groundTruth.pt')

        f.close()  # Close the text file
        # Color-coding the minimum RMSE value at each timeframe in the prediction horizon
        format1 = book.add_format({'bg_color': '#B7DBFF'})
        for rNum in range(26):
            sheet.conditional_format('C{i}:G{j}'.format(i=rNum + 1, j=rNum + 1),
                                     {"type": "bottom", "value": 1, "format": format1})
        book.close()

if __name__ == '__main__':
    names = ['20']
    evaluate = Evaluate()
    for epoch in names:
        evaluate.main(name=epoch, val=False)
