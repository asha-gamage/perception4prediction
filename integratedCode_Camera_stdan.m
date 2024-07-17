%% Script Description
% This script takes in the Camera sensor readings along with the ego car global coordinates and calculates 
% the global x,y coordinates of the traffic cars then to be used to produce
% the .mat file to be fed into the CS-LSTM model to predict the trajectory of the target car 
% Updated the code to determine the detection of an object by checking the Confidence for that object being ‘0’, i.e. not detected
% Updated the code to include the ground truth tracks (directly read from CM), such that the derivation of 'fut' which is the 
% ground truth trajectory positions for the prediction horizon of 5sec to the future to be used for the RMSE calculation is 
% always available (as opposed to potentially missing detections if the sensor based future frames are used for the 'fut' calculation.

close all; % close all the figures
clear;

% This section calculates the equation of the line the nose of the vehicle is pointing at a given time.

Speed = 5;
Dir = 'Right';
Road = 'Straight';
Param = 'Range';
Scenario = '_infront of Lead_';
Data_impute = false;
Spd = num2str(Speed);

switch Dir
    case 'Right'
        folder = 'RLC';
    case 'Left'
        folder = 'LLC';
end    

switch Road
    case 'Straight'
        rd = 'StraightRd';
    case 'US101'
        rd = 'US101_original';
    case 'HW'
        rd = 'HW Freeway';
    case 'Curve'
        rd = 'Road Curve';
end

switch Param
    case 'Range'
        unit = 'm';
    case 'FoV'
        unit = 'deg';
end

% Create an Excel file to record the number of timeframes a traffic car gets detected by the Radar
filename = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\cameraData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\No.of Detections.xlsx'];
names = {Param,"Target","Lead to Ego","Lead to Target","NIO","Lexus"};
% writecell(names,filename);

% for a = 30:30:180    
for a = 25:25:150
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Camera\',Param,'\RelLonVel_',Spd,'kph\VerFoV@15deg\Scenario_',rd,'_',folder,'_2023_HorFoV (',num2str(a),').erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Camera\Range\RelLonVel_',Spd,'kph\Scenario_',rd,'_',folder,'_4TS','_Range_',num2str(a),'.erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Camera\Range\RelLonVel_',Spd,'kph\Scenario_US101_LC_',Dir,Scenario,Param,'_',num2str(a),'.erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Camera\',Param,'\RelLonVel_',Spd,'kph\Scenario_',Road,'_LC_',Dir,'_',Param,'_',num2str(a),'.erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Camera\',Param,'\RelLonVel_',Spd,'kph\VerFoV@15deg\Scenario_US101_LC_',Dir,Scenario,'HorFoV (',num2str(a),').erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Camera\',Param,'\RelLonVel_',Spd,'kph\VerFoV@15deg\Scenario_',rd,'_LC_',Dir,'_HorFoV (',num2str(a),').erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Camera\',Param,'\RelLonVel_',Spd,'kph\VerFoV@15deg\Scenario_',Road,'_LC_',Dir,'_infront of ego_Hor',Param,' (',num2str(a),').erg'];
    readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Camera\',Param,'\RelLonVel_',Spd,'kph\Scenario_',rd,'_LC_',Dir,'_',Param,'_',num2str(a),'.erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Camera\',Param,'\RelLonVel_',Spd,'kph\Scenario_',rd,'_',folder,'_2023_Range_',num2str(a),'.erg'];
    dFile = cmread(readPath);
    ego_xFr1 = dFile.Car_Fr1_tx.data; % Ground truth longitudinal distance
    ego_yFr1 = dFile.Car_Fr1_ty.data; % Ground truth lateral distance
    ego_velX = dFile.Car_Fr1_vx.data; % X componant of velocity of the ego vehicle
    ego_velY = dFile.Car_Fr1_vy.data; % Y componant of velocity of the ego vehicle
    ego_acclX = dFile.Car_Fr1_ax.data; % X componant of acceleration of the ego vehicle   
    ego_acclY = dFile.Car_Fr1_ay.data; % X componant of acceleration of the ego vehicle   
    Time = dFile.Time.data;
    rdRefAng = asin(dFile.RdVector_y.data);
    yawAng = dFile.Car_Yaw.data; % Ground truth yaw angle
    yawAng = round(yawAng*100)/100; % Convert the yaw angle to 2 decimal point representation
    camPos = 2.5; % Position of the camera sensor on the vehicle
    ego_x = ego_xFr1 + camPos.*cos(yawAng); % Ground truth x coordinate of camera position
    ego_y = ego_yFr1 + camPos.*sin(yawAng); % Ground truth y coordinate of camera position
    
    % Traffic car detection and ground truth data
    numTraffic = max(dFile.Sensor_Camera_Camera1_nObj.data); % Maximum number of objects detected by the vision sensor
    tt = struct2cell(dFile); % Convert the dFile structure to a cell, to enable reading the strings
    
    % This section extracts the required camera measurements and ground-truth data to subsequently calculate x, y coordinates 
    % of the traffic cars 
    detdCars = [];   
    for i = 1:numTraffic
        traffic = ['Sensor.Camera.Camera1.Obj.' num2str(i-1) '.ObjID'];
        for q = 1:length(tt)
            if convertCharsToStrings(tt{q}.name) == convertCharsToStrings(traffic)
                objIDs = unique(tt{q}.data);
                detdCars = [detdCars,objIDs];
            end
        end
    end
    detdCars = detdCars(find(detdCars>=16000000));
    detdCars = unique(detdCars);
    measuredData = cell(max(detdCars)-16000000+1,2); 
    measuredData(cellfun(@isempty,measuredData)) = {zeros(size(Time))};
    
    for i = 1:numTraffic 
        cam_Idx = ['Sensor.Camera.Camera1.Obj.' num2str(i-1)]; 
        Trf_ID = [cam_Idx '.ObjID'];
        meas_X = [cam_Idx '.MBR.BL_X'];
        meas_TRy = [cam_Idx '.MBR.TR_Y'];
        meas_BLy = [cam_Idx '.MBR.BL_Y'];       
        Detect = [cam_Idx '.Confidence'];
        
        % Iterate through all the records in the cell to fill the data cell with filtered data
        for q = 1:length(tt)
            if convertCharsToStrings(tt{q}.name) == convertCharsToStrings(Trf_ID)
                objIDs = unique(tt{q}.data);
                objIDs = objIDs(find(objIDs>=16000000));
                
                for obj = 1:length(objIDs)
                    tarIdx = find(tt{q}.data == objIDs(obj));
                    
                    GT_traffic_X = ['Traffic.T0' num2str(objIDs(obj)-16000000) '.tx'];
                    GT_traffic_Y = ['Traffic.T0' num2str(objIDs(obj)-16000000) '.ty'];
                    GT_traffic_Xvel = ['Traffic.T0' num2str(objIDs(obj)-16000000) '.v_0.x'];
                    GT_traffic_Yvel = ['Traffic.T0' num2str(objIDs(obj)-16000000) '.v_0.y'];
                    GT_traffic_Xaccl = ['Traffic.T0' num2str(objIDs(obj)-16000000) '.a_0.x'];
                    GT_traffic_Yaccl = ['Traffic.T0' num2str(objIDs(obj)-16000000) '.a_0.y'];
                    laneID = ['Traffic.T0' num2str(objIDs(obj)-16000000) '.Lane.Act.LaneId'];
                    
                    for k = 1:length(tt)
                        % Fill the measuredData cell structure with the required data only.
                        if convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_X)
                            measuredData{objIDs(obj)-16000000+1,1}(1,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_TRy)
                            measuredData{objIDs(obj)-16000000+1,1}(2,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_BLy)
                            measuredData{objIDs(obj)-16000000+1,1}(3,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(Detect)
                            measuredData{objIDs(obj)-16000000+1,1}(4,tarIdx) = tt{k}.data(tarIdx);                            
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_X)
                            measuredData{objIDs(obj)-16000000+1,2}(1,:) = tt{k}.data;
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Y)
                            measuredData{objIDs(obj)-16000000+1,2}(2,:) = tt{k}.data;
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Xvel)
                            measuredData{objIDs(obj)-16000000+1,2}(3,:) = tt{k}.data(1:end);  
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Yvel)
                            measuredData{objIDs(obj)-16000000+1,2}(4,:) = tt{k}.data(1:end); 
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Xaccl)
                            measuredData{objIDs(obj)-16000000+1,2}(5,:) = tt{k}.data(1:end); 
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Yaccl)
                            measuredData{objIDs(obj)-16000000+1,2}(6,:) = tt{k}.data(1:end);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(laneID)
                            measuredData{objIDs(obj)-16000000+1,2}(7,:) = tt{k}.data;
                        end
                    end          
                end          
            end
        end
    end
    
    % Create a cell for the calculated x, y coordinates of the traffic cars
    measurements = cell(length(measuredData),8);
    
    for j = (detdCars)-16000000+1
        for i = 1:length(measuredData{j,1}) 
            if  ~isempty(measuredData{j,1})
                if ~(measuredData{j,1}(4,i) == 0) 
                    averageY = measuredData{j,1}(2,i) + 0.5*(measuredData{j,1}(3,i)-measuredData{j,1}(2,i)); % Calculate the lateral distance to the centre of the rectangle
                    if yawAng(i) == 0 % if the ego vehicle trajectory is straight inline with the road center line (since road center line is set to the x-axis of the earth's fixed system).
                        measured_x = ego_x(i) + measuredData{j,1}(1,i);
                        measured_y = ego_y(i) + averageY; % assuming the measurements to the right of the line are negetive
                        dis_x = measuredData{j,1}(1,i);
                        dis_y = averageY;
                    else
                        theta = atan(averageY/ measuredData{j,1}(1,i)); 
                        Dist = sqrt(measuredData{j,1}(1,i)^2 + averageY^2);
                        alpha = yawAng(i) + theta;
                        del_x = cos(alpha)* Dist;
                        del_y = sin(alpha)* Dist;
                        measured_x = ego_x(i) + del_x;
                        measured_y = ego_y(i) + del_y;
                        dis_x = del_x;
                        dis_y = del_y;
                        
                    end
                else
                    measured_x = 0;
                    measured_y = 0;
                    dis_x = 0;
                    dis_y = 0;
                end
            else
                measured_x = 0;
                measured_y = 0;
                dis_x = 0;
                dis_y = 0;
            end
            measurements{j,1} = [measurements{j,1}, measured_x]; 
            measurements{j,2} = [measurements{j,2}, measured_y];
            measurements{j,3} = [measurements{j,3}, dis_x]; 
            measurements{j,4} = [measurements{j,4}, dis_y];   
        end       
        relV_x = diff(measurements{j,3})./diff(Time);
        relA_x = diff(relV_x)./diff(Time(2:end));
        relV_x = [0,relV_x];
        relA_x = [0,0,relA_x];
        measurements{j,5} = relV_x + ego_velX; 
        measurements{j,7} = relA_x + ego_acclX; 
        relV_y = diff(measurements{j,4})./diff(Time);
        relA_y = diff(relV_y)./diff(Time(2:end));
        relV_y = [0,relV_y];
        relA_y = [0,0,relA_y];
        measurements{j,6} = relV_y + ego_velY; % y_velocity component of the detected traffic object
        measurements{j,8} = relA_y + ego_acclY; % y_acceleration component of the detected traffic object     
        
        
        %% Data Imputation
        % This section fills the missing detection timeframes with simple interpolated data to facilitate the correct operation of the trajectory prediction model, STADN
        if Data_impute            
            dim = size(measurements);
            rows = dim(1,1);
            cols = dim(1,2);
            for r = 1:rows
                for c = 1:cols
                    senData = measurements{r,c};
                    non0start = find(senData ~= 0, 1, 'first');
                    non0end = find(senData ~= 0, 1, 'last');
                    senData = senData(non0start:non0end);
                    senData(senData==0)=NaN; 
                    xDataImputed = fillmissing(senData,'linear'); 
                    measurements{r,c}(non0start:non0end) = xDataImputed; 
                end
            end
        end
        %% Differences between the ground-truth coordinates Vs the calculated coordinates based on camera measurements of the target car
        if  ~isempty(measuredData{j,1})
            index = (measurements{j,1} ~= 0);
            calcCoordinate_x = measurements{j,1}(index);
            gTruth_x = measuredData{j,2}(1,:);
            gTruth_x_filtered= gTruth_x(index);
            diff_x = gTruth_x_filtered - calcCoordinate_x;
            
            compare_x = [gTruth_x_filtered', calcCoordinate_x', (gTruth_x_filtered - calcCoordinate_x)'];
            
            calcCoordinate_y = measurements{j,2}(index);
            gTruth_y = measuredData{j,2}(2,:);
            gTruth_y_filtered= gTruth_y(index);
            diff_y = gTruth_y_filtered - calcCoordinate_y;
            
            compare_y = [gTruth_y_filtered', calcCoordinate_y', (gTruth_y_filtered - calcCoordinate_y)'];
            
            % Compare the calculated Vs groundtruth X_velocity component
            index = (measurements{j,5} ~= 0); 
            calcVelX = measurements{j,5}(index);
            gTruth_VelX = measuredData{j,2}(3,:);
            gTruth_VelX_filtered = gTruth_VelX(index);
            diff_velX = gTruth_VelX_filtered - calcVelX;
            
            % Compare the calculated Vs groundtruth Y_velocity component
            calcVelY = measurements{j,6}(index);
            gTruth_VelY = measuredData{j,2}(4,:);
            gTruth_VelY_filtered = gTruth_VelY(index);
            diff_velY = gTruth_VelY_filtered - calcVelY;    
            
            % Compare the calculated Vs groundtruth X_acceleartion component
            index = (measurements{j,7} ~= 0); % get the indices of the imputed data
            calcAccX = measurements{j,7}(index);
            gTruth_AccX = measuredData{j,2}(5,:);
            gTruth_AccX_filtered = gTruth_AccX(index);
            diff_accX = gTruth_AccX_filtered - calcAccX;        
            
            % Compare the calculated Vs groundtruth Y_acceleartion component
            calcAccY = measurements{j,8}(index);
            gTruth_AccY = measuredData{j,2}(6,:);
            gTruth_AccY_filtered = gTruth_AccY(index);
            diff_accY = gTruth_AccY_filtered - calcAccY;   
            
            %% Plotting the difference between measurement-based Vs ground truth coordinates
            % close all;
            % Create a figure
            TrfID = num2str(j-1);
            if TrfID == '0'
                Title_1 = ['Ground Truths Vs Camera Measurements at ',Param,' ', num2str(a),' for Target Car',' ',rd];
            else
                Title_1 = ['Ground Truths Vs Camera Measurements at ',Param,' ', num2str(a),' for Surround Car_', TrfID,' ',rd];
            end          
            figMetrics = figure('Name',Title_1);           
            maxWidth = 1150;
            pos = figMetrics.Position;
            width = pos(3);
            figMetrics.Position = [pos(1)-(maxWidth-width)/2 pos(2) maxWidth pos(4)];
            
            % Plot the camera measurements-based x coordinates
            xCord_camera = subplot(1,2,1);
            plot(calcCoordinate_x,'.');
            hold on
            plot(gTruth_x_filtered, '.');
            hold off
            
            ylabel(xCord_camera,'x coordinate');
            xlabel(xCord_camera,'Point');
            title(xCord_camera,'Camera-based Vs Groundtruths for X coordinates');
            grid(xCord_camera,'on');
            legend ('Camera detections','GroundTruth','Location','northwest');
            
            % Plot the camera measurements-based y coordinates
            yCord_camera = subplot(1,2,2);
            plot(calcCoordinate_y,'.');
            hold on
            plot(gTruth_y_filtered, '.');
            hold off
            ylabel(yCord_camera,'y coordinate');
            xlabel(yCord_camera,'Point');
            title(yCord_camera,'Camera-based Vs Groundtruths for Y coordinates');
            grid(yCord_camera,'on');
            legend ('Camera detections','GroundTruth','Location','northeast');
                                   
            % Define path for saving the figures
            FolderName = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\cameraData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\graphs'];
            %[~, file]  = fileparts(filename);  % Remove extension
            saveas(gca, fullfile(FolderName, [Title_1, '.png']) ) % Append
            close all;
            
            % Plot the camera detection against the ground truth 
            figure;
            if TrfID == '0'
                Title_2 = ['Ground Truths Vs Camera based Detections at ',Param,' ', num2str(a),' for Target Car'];
            else
                Title_2 = ['Ground Truths Vs Camera based Detections at ',Param,' ', num2str(a),' for Surround Car_', TrfID];
            end
            
            plot(calcCoordinate_x,calcCoordinate_y,'^','MarkerSize',1.5, 'LineWidth', 1.5); 
            hold on 
            plot(gTruth_x_filtered,gTruth_y_filtered,'-','LineWidth', 1); 
            hold off
            ylabel('y coordinate');
            xlabel('x coordinate');
            if TrfID == '0'
                title('Camera Detections Vs Ground-Truths for the Target Car ');
            else
                title(['Camera Detections Vs Ground-Truths for the Surround Car ', TrfID]);
            end
            grid('on');
            legend ('Camera Detections','Ground-Truth','Location','northeast');
            
            % Define path for saving the figure
            FolderName = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\cameraData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\graphs'];
            %[~, file]  = fileparts(filename);  % Remove extension
            saveas(gca, fullfile(FolderName, [Title_2, '.png']) ) % Append
            close all;
            
            %% Plotting the X, Y components of velocity from measurement-based Vs ground truths
            % close all;
            % Create a figure
            TrfID = num2str(j-1);
            if TrfID == '0'
                Title_3 = ['Ground Truths Velocity Vs Camera Measurements at ',Param,' ', num2str(a),'m for Target car',' ',rd];
            else
                Title_3 = ['Ground Truths Velocity Vs Camera Measurements at ',Param,' ', num2str(a),'m for Surround Car_', TrfID,' ',rd];
            end
            figMetrics = figure('Name',Title_3);
            % This is the maximum figure width that can be used for publishing without clipping the subplots
            maxWidth = 1150;
            pos = figMetrics.Position;
            width = pos(3);
            figMetrics.Position = [pos(1)-(maxWidth-width)/2 pos(2) maxWidth pos(4)];
            % plot the radar measurements-based velocity
            velX_camera = subplot(1,2,1);
            plot(calcVelX,'.');
%             axis ([0 600 5 20]);
            
            hold on
            plot(gTruth_VelX_filtered, '.');
            hold off            
            ylabel(velX_camera,'X velocity');
            xlabel(velX_camera,'Point');
            title(velX_camera,'Camera-based Vs Ground-truths for X-Velocity');
            grid(velX_camera,'on');
            legend ('Camera detections','GroundTruth','Location','northwest');
            
            % plot the radar measurements-based y coordinates
            velY_camera = subplot(1,2,2);
            plot(calcVelY,'.');
%             axis ([0 600 -25 0]);
            
            hold on
            plot(gTruth_VelY_filtered, '.');
            hold off
            ylabel(velY_camera,'Y velocity');
            xlabel(velY_camera,'Point');
            title(velY_camera,'Camera-based Vs Ground-truths for Y-Velocity');
            grid(velY_camera,'on');
            legend ('Camera detections','GroundTruth','Location','northeast');

            % Plot the velocities
            figure;
            if TrfID == '0'
                Title_3 = ['Ground Truths Vs Camera-based Velocity at ',Param,' ', num2str(a),' for Target Car'];
            else
                Title_3 = ['Ground Truths Vs Camera-based Velocity at ',Param,' ', num2str(a),' for Surround Car_', TrfID];
            end
            V_cam = sqrt(calcVelX.^2+calcVelY.^2);
            V_GT = sqrt(gTruth_VelX_filtered.^2+gTruth_VelY_filtered.^2);

            plot(V_cam, '^','MarkerSize',1.5, 'LineWidth', 1.5);
%             axis ([0 600 15 30]);
            hold on
            plot(V_GT,'-','LineWidth', 1);
            hold off

            ylabel('Velocity');
            xlabel('point');
            if TrfID == '0'
                title('Camera-based Vs Ground-Truths velocity for the Target Car ');
            else
                title(['Camera-based Vs Ground-Truths velocity for the Surround Car ', TrfID]);
            end
            legend ('Camera Detections','Ground-Truth','Location','northeast');
            
            % Define path for saving the figure
            FolderName = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\cameraData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\graphs'];
            %[~, file]  = fileparts(filename);  % Remove extension
            saveas(gca, fullfile(FolderName, [Title_3, '.png']) ) % Append
            close all;    
            
            % Plotting the X, Y components of acceleration from measurement-based Vs ground truths
            % close all;
            % Create a figure
            TrfID = num2str(j-1);
            if TrfID == '0'
                Title_4 = ['Ground Truths Acceleration Vs Camera Measurements at ',Param,' ', num2str(a),'m for Target car',' ',rd];
            else
                Title_4 = ['Ground Truths Acceleration Vs Camera Measurements at ',Param,' ', num2str(a),'m for Surround Car_', TrfID,' ',rd];
            end
            figMetrics = figure('Name',Title_4);
            % This is the maximum figure width that can be used for publishing without clipping the subplots
            maxWidth = 1150;
            pos = figMetrics.Position;
            width = pos(3);
            figMetrics.Position = [pos(1)-(maxWidth-width)/2 pos(2) maxWidth pos(4)];
            % plot the camera measurements-based velocity
            accelX_camera = subplot(1,2,1);
            plot(calcAccX,'.');
%             axis ([0 600 5 40]);
            hold on
            plot(gTruth_AccX_filtered, '.');
            hold off            
            ylabel(accelX_camera,'X acceleration');
            xlabel(accelX_camera,'Point');
            title(accelX_camera,'Camera-based Vs Ground-truths for X-Acceleration');
            grid(accelX_camera,'on');
            legend ('Camera detections','GroundTruth','Location','northwest');
            
            % plot the camera measurements-based acceleration
            accelY_camera = subplot(1,2,2);
            plot(calcAccY,'.');
            hold on
            plot(gTruth_AccY_filtered, '.');
            hold off
            ylabel(accelY_camera,'Y acceleration');
            xlabel(accelY_camera,'Point');
            title(accelY_camera,'Camera-based Vs Groundtruths for Y-Acceleration');
            grid(accelY_camera,'on');
            legend ('Camera detections','GroundTruth','Location','northeast');

            figure;
            if TrfID == '0'
                Title_4 = ['Ground Truths Vs Camera-based Acceleration at ',Param,' ', num2str(a),' for Target Car'];
            else
                Title_4 = ['Ground Truths Vs Camera-based Acceleration at ',Param,' ', num2str(a),' for Surround Car_', TrfID];
            end
            A_cam = sqrt(calcAccX.^2+calcAccY.^2);
            A_GT = sqrt(gTruth_AccX_filtered.^2+gTruth_AccY_filtered.^2);

            plot(A_cam, '^','MarkerSize',1.5, 'LineWidth', 1.5);
            hold on
            plot(A_GT,'-','LineWidth', 1);
            hold off
            ylabel('Acceleration');
            xlabel('point');
            if TrfID == '0'
                title('Camera-based Vs Ground-Truths acceleration for the Target Car ');
            else
                title(['Camera-based Vs Ground-Truths acceleration for the Surround Car ', TrfID]);
            end
            legend ('Camera Detections','Ground-Truth','Location','northeast');
            % Define path for saving the figure
            FolderName = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\cameraData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\graphs'];
            %[~, file]  = fileparts(filename);  % Remove extension
            saveas(gca, fullfile(FolderName, [Title_4, '.png']) ) % Append
            close all;            
        end
    end

    %% Script for generating the traj data file for the STDAN model using CM generated data from the Camera measurements.

    % Read the data from the saved data file from CM
    % trajCM is the trajectory details of the Target car.

    surCars = [];
    for t = (detdCars)-16000000+1
    %for t = 1:(detdCars)-16000000+1
        if  ~isempty(measuredData{t,1})% Check if the vehicle is detected
            relV_x = diff(measurements{t,3})./diff(Time);
            trajTr = [10*Time', measurements{t,2}', measurements{t,1}',  measurements{t,5}',  measurements{t,6}', measurements{t,7}', measurements{t,8}', measuredData{t,2}(5,:)',ones(length(Time),1)*2];% [time, local x, local y, v_Vel, v_Acc, lane id, v_type] correspond to the Target car (always given traffic name 'T00' in CM)
            vehID = ones(length(trajTr),1)*t+15; % Generate the global vehicle IDs as in CM by multiplying the ones array by 16(In CM:16000000)
            trajTr = single([vehID, trajTr]); %  Format the first 6 fields of the matrix and convert to single precision
            [~,ia,~] = unique(measuredData{t,2}(1:3,:)','rows', 'stable'); % Remove the time frames matching the static poisitional data records from groundtruth data records
            trajTr = trajTr(ia,:);
            if t==1 %  This is the target car
                trajTar = [trajTr, rdRefAng(ia)']; 
                non0start = find(trajTr(:,3) ~= 0, 1, 'first'); 
                non0end = find(trajTr(:,3) ~= 0, 1, 'last'); 
                trajtar = trajTr(non0start:non0end,:); 
            else                 
            surCars = [surCars;trajTr]; % For surround cars, impact of occlusions are accounted for since the no data points are not removed
            end
        end
    end
    
    % Create an array of all the trajectory data of the ego vehicle
    trajEgo = [ones(length(Time'),1)*15,10*Time', ego_yFr1', ego_xFr1', ego_velX', ego_velY', ego_acclX', ego_acclY', dFile.Car_Road_Lane_Act_LaneId.data',ones(length(Time'),1)*2]; % [vehID, time, local x, local y, v_Vel, v_Acc, landID, v_type]
    [~,ia,~] = unique(trajEgo(:,3:4),'rows', 'stable'); % Remove the static poisition data recording at the end of the scenario
    trajEgo = single(trajEgo(ia,:)); % Use 'ia' to index into and retrieve the rows that have unique combinations of elements in the 4th and 5th columns.
                                     % and convert the values into single precision.
    trajSur = [surCars;trajEgo];
    trajAll = [trajSur;trajtar];

    [traj, tracks] = CMinputs(trajTar, trajSur, trajAll);
    
    % Populate an Excel file to log the number of detections of traffic cars by the Radar sensor and a graph to plot the detections of
    % traffic during the simulation period   
    figure;
    ax = axes();
    
    obsVehIDs=find(~cellfun(@isempty,tracks)); 
    for i = obsVehIDs
        data = tracks{i};
        cordData = data(2:3,:);
        columnsWithAllZeros = all(cordData == 0);
        detected = data(:, ~columnsWithAllZeros);
        numDetected = length(detected);
        D{1,i-14} = numDetected;
        D{1,1} = [num2str(a),unit];        
        y = ~columnsWithAllZeros*(i-15);
        y(y==0) = NaN;
        detectStart = ones(1,data(1,1)-1)*NaN;
        y = [detectStart,y];        
        plot(y,'^','MarkerSize',1.5, 'LineWidth', 1.5);
        hold on
    end
    
    title(['Traffic Detection by the Camera at ',Param,' ',num2str(a),unit]);
    Title_3 = ['\Traffic Detection by the Camera at ',Param,' ',num2str(a),unit];
    xlabel('Simulation Time');
%     ylim([0,5]) 
    ylim([0,6]) 
%     set(gca,'ytick', 0:1:5);
    set(gca,'ytick', 0:1:6);
%     yticklabels({' ','Target car','Lead to Ego','Lead to Target', 'NIO'});
    yticklabels({' ','Target car','Lead to Ego','Lead to Target', 'NIO', 'Lexus'});
    %set(gca,'fontweight','bold')    
%     xticks(0:50:300);
    xticks(0:50:600);
    ax.XAxis.MinorTick = 'on';
%     ax.XAxis.MinorTickValues = 0:10:300;
    ax.XAxis.MinorTickValues = 0:10:600;
%     xticklabels({'0s','5s','10s', '15s','20s','25s','30s'})
    xticklabels({'0s','5s','10s', '15s','20s','25s','30s','35s','40s', '45s','50s','55s','60s'})
        
    lcDet = traj(:,10);
    lcBound = find(diff(lcDet));
    if length(lcBound)==1
        xline(traj(lcBound+1,3),'-b',{'Lane Change Start'},'LineWidth',2);        
    else
       for b = 1:length(lcBound)/2 % b is number of lane changes in the scenario
            x1 = traj(lcBound(2*b-1)+1,3);
            x2 = traj(lcBound(2*b)+1,3);
            yl = ylim;
            xBox = [x1, x1, x2, x2, x1];
            yBox = [yl(1), yl(2), yl(2), yl(1), yl(1)];
            patch(xBox, yBox, 'white','FaceColor', 'blue','EdgeColor','none','FaceAlpha', 0.1);
            text(double(x1+(x2-x1)/2),1.5,'Lane Change Window','Fontsize',13,'Rotation',90);   %'Color','blue',
        end 
    end   
    saveas(gcf,[FolderName,'\',Title_3,'.png'])
    hold off
    close all;
    
%     writecell(D,filename,'WriteMode','append');
    
    %% Script for generating the traj data file for the CS pooling model using CM based ground-truth data

% Read the data from the saved data file from CM
surCarsgT = [];
for t = (detdCars)-16000000+1
    trajTrGT = [10*Time', measuredData{t,2}(2,:)', measuredData{t,2}(1,:)', measuredData{t,2}(3,:)', measuredData{t,2}(4,:)', measuredData{t,2}(5,:)', measuredData{t,2}(6,:)', measuredData{t,2}(7,:)', ones(length(Time),1)*2];% correspond to the Target car (always given traffic name 'T00' in CM)
    vehID = ones(length(trajTrGT),1)*(t+15); % Generate the global vehicle IDs as in CM by multiplying the ones array by 16(In CM:16000000)
    trajTrGT = single([vehID, trajTrGT]); %  Format the first 8 fields of the matrix and convert to single precision
    [~,ia,~] = unique(trajTrGT(:,3:4),'rows', 'stable'); % Remove the static poisition data recording at the end of the scenario
    trajTrGT = trajTrGT(ia,:);
    if t==1
        trajTarGT = [trajTrGT, rdRefAng(ia)'];
    else        
        surCarsgT = [surCarsgT;trajTrGT];
    end
end

trajSurGT = [surCarsgT; trajEgo]; % Combine the surround vehicles together
trajAllGT = [trajSurGT; trajTarGT(:,1:end-1)];

[trajGndT, tracksGndT] = CMinputs(trajTarGT, trajSurGT, trajAllGT);
    
    %% Save mat files:
    %disp('Saving mat files...')
    savePath = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\cameraData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\matFiles\',rd,'_',Param,'_',num2str(a)];
%     savePath = ['C:\Users\gamage_a\Documents\Python\conv-social-pooling-master\',rd,'\cameraData\manoeuvre_',folder,'\Range\relVel_',Spd,'kph\matFiles\',rd,'_Range_',num2str(a),' occluded'];
    save(savePath,'traj','tracks','tracksGndT')
end 

%% Script for generating the traj data file for the STDAN model using CM based ground-truth data

% Read the data from the saved data file from CM
surCarsgT = [];
for t = (detdCars)-16000000+1
    trajTrGT = [10*Time', measuredData{t,2}(2,:)', measuredData{t,2}(1,:)', measuredData{t,2}(3,:)', measuredData{t,2}(4,:)', measuredData{t,2}(5,:)', measuredData{t,2}(6,:)', measuredData{t,2}(7,:)', ones(length(Time),1)*2];% correspond to the Target car (always given traffic name 'T00' in CM)
    vehID = ones(length(trajTrGT),1)*(t+15); % Generate the global vehicle IDs as in CM by multiplying the ones array by 16(In CM:16000000)
    trajTrGT = single([vehID, trajTrGT]); %  Format the first 8 fields of the matrix and convert to single precision
    [~,ia,~] = unique(trajTrGT(:,3:4),'rows', 'stable'); % Remove the static poisition data recording at the end of the scenario
    trajTrGT = trajTrGT(ia,:);
    if t==1
        trajTarGT = [trajTrGT, rdRefAng(ia)'];
    else        
        surCarsgT = [surCarsgT;trajTrGT];
    end
end

trajSurGT = [surCarsgT; trajEgo]; % Combine the surround vehicles together
trajAllGT = [trajSurGT; trajTarGT(:,1:end-1)];

[traj, tracks] = CMinputs(trajTarGT, trajSurGT, trajAllGT);

%% Save mat files:
%disp('Saving mat files...')
savePath = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\cameraData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\matFiles\',rd,'_groundTruth'];
% savePath = ['C:\Users\gamage_a\Documents\Python\conv-social-pooling-master\',rd,'\cameraData\manoeuvre_',folder,'\Range\relVel_',Spd,'kph\matFiles\',rd,'_groundTruth occluded'];
save(savePath,'traj','tracks','tracksGndT');

function [traj, tracks] = CMinputs(trajCM, trajTr, trajAll)
% Modified N.Deo's code to determine the lateral and longitudinal
% maneuvers: Same approach followed as implemented when using the NGSIM data.

for k = 1:length(trajCM(:,1))        
    time = trajCM(k,2);  
    vehId = trajCM(k,1);
    lane = trajCM(k,9);
    ind = find(trajCM(:,2)==time);
    ind = ind(1); % Find the index of the vehicle trajectory where the frame ID (column 3) matches the 'time'.I.e. find the index of 
    % the exact track from the list of tracks for a vehicle: vehid
    % find(): Find indices of nonzero elements        

    % Get lateral maneuver:
    ub = min(size(trajCM,1),ind+40);% Upper bound is calculated by checking whether the index at each record has 40 
                                     %(0.5*8sec duration used for observation and prediction) more Frame_IDs to the future and taking the lowest timeFrame
                                     % size(X,1) returns the number of rows of X and size(X,[1 2]) returns a row vector containing the number of rows & columns.
    lb = max(1, ind-40);% Lower bound is calculated by checking whether the index at each record has 40 more Frame_IDs to the past and taking 
                                    % the highest Frame_ID
    if trajCM(ub,9)>trajCM(ind,9) || trajCM(ind,9)>trajCM(lb,9)% future lane Id > current lane Id OR current lane Id > past lane Id 
        trajCM(k,12) = 2;% Categorise as 'Right Lane-change' and adds it to a new column_7
    elseif trajCM(ub,9)<trajCM(ind,9) || trajCM(ind,9)<trajCM(lb,9)
        trajCM(k,12) = 3;% Left Lane-change
    else
        trajCM(k,12) = 1;% Lane Keep
    end        

    % Get longitudinal maneuver:
    ub = min(size(trajCM,1),ind+50);% Upper bound is calculated by checking whether the index at each record has 50 more frames(5s to the future)to 
                                     % the future and taking the lowest duration

    lb = max(1, ind-30);% Lower bound is calculated by checking whether the index at each record has 30 frames to the past or checking if at the start 
                        % of the recording
    if ub==ind || lb ==ind || trajCM(ub,4)==0 ||trajCM(lb,4)==0 % If current index is the start OR the end of the recording OR there's no longitudinal 
                                                                % reading (no measurement from the sensor) available....
        trajCM(k,12) =1; % longitudinal maneuver is categorised as 'Normal speed' and adds it to a new column_8. This is questionable due to occlusions 
                        % when using camera
    else
        vHist = (trajCM(ind,4)-trajCM(lb,4))/(ind-lb);% Historical velocity calculated by dividing the longitudinal distance between 
                                                        % current and lower bound time frames
        vFut = (trajCM(ub,4)-trajCM(ind,4))/(ub-ind);% Future velocity calculated by dividing the longitudinal distance between 
                                                       % current and lower bound time frames
        if vFut/vHist < 0.8% vehicle to be performing a braking maneuver if it’s average speed over the prediction horizon is less 
            % than 0.8 times its speed at the time of the prediction
            trajCM(k,13) = 2;% Braking and adds it to a new column_8
        elseif vFut/vHist > 1.25
            trajCM(k,13) = 3;% Accel
        else
            trajCM(k,13) = 1;
        end
    end

    % Get grid locations:
    frameEgo = trajTr(and((trajTr(:,2) == time),(trajTr(:,9) == lane)),:);
    frameL = trajTr(and((trajTr(:,2) == time),(trajTr(:,9) == lane-1)),:);
    frameR = trajTr(and((trajTr(:,2) == time),(trajTr(:,9) == lane+1)),:);        
    if ~isempty(frameL)
        for l = 1:size(frameL,1)
            y = frameL(l,4)-trajCM(k,4);
            if abs(y) < 27.432 % 90feet distance boundary
                gridInd = 1+round((y+27.432)/4.572); % Filling the first column of the 13x3 spatial grid
                trajCM(k,13+gridInd) = frameL(l,1);
            end
        end
    end
    for l = 1:size(frameEgo,1)
        y = frameEgo(l,4)- trajCM(k,4);
        if abs(y) < 27.432 && y~=0
            gridInd = 14+round((y+27.432)/4.572);% 14 come from 90ft/15ft to the front and back of the ego car +1 (13) allocated to the frameL earlier.
            trajCM(k,13+gridInd) = frameEgo(l,1);
        end
    end
    if ~isempty(frameR)
        for l = 1:size(frameR,1)
            y = frameR(l,4)-trajCM(k,4);
            if abs(y) < 27.432
                gridInd = 27+round((y+27.432)/4.572);% 27 comes from (14 + 13) as above.
                trajCM(k,13+gridInd) = frameR(l,1);
            end
        end
    end        
end

tracksCM = {};
carIds = unique(trajAll(:,1)); % Create an array containing the unique vehicleIDs
for l = 1:length(carIds)
vehtrack = trajAll(trajAll(:,1) == carIds(l),2:10)'; % ***Need to take the transpose for my research.
% Iterate over the unique vehicleIDs and get the frameID, localX and localY and
% transpose the matrix to a 3 by length of frame numbers captured
tracksCM{1,carIds(l)} = vehtrack; % create a cell with references; DatasetID as the row and the vehicle ID as the column
end

% % Filterout trajectories for the target car (for prediction)
% trajTar = trajAll(find(trajAll(:,1)==16),:);
% trajTar = [trajTar, RdRefAng]; 
%     
% % Remove only the leading and lagging zeros to the sensor captured data for the target car
% non0start = find(trajTar(:,3) ~= 0, 1, 'first'); % index of first non-zero data recording
% non0end = find(trajTar(:,3) ~= 0, 1, 'last'); % index of first non-zero data recording
% trajTar = trajTar(non0start:non0end,:); % Limit the predictions on the time window the sensor had captured the target car's trajectory including occlusions        

% Remove time steps with no data for x, y coordinates 
idx = find(trajCM(:,4));
trajCM = trajCM(idx,:);

%% Filter edge cases: 
% Since the model uses 3 sec of trajectory history for prediction, the initial 3 seconds of each trajectory is not used for training/testing
%disp('Filtering edge cases...')
% if GT
indsCM = zeros(size(trajCM,1),1);
for k = 1: size(trajCM,1)
    t = trajCM(k,2);
    if tracksCM{1,trajCM(k,1)}(1,31) <= t && tracksCM{1,trajCM(k,1)}(1,end)>t+1
        indsCM(k) = 1;
    end
end

trajCM = trajCM(find(indsCM),:);%  find(): Find indices of nonzero elements

% Add dataset Id column
traj = [ones(size(trajCM,1),1),trajCM]; % Add a Dataset ID as first column as expected by the PyTorch model
tracks = tracksCM;
end


