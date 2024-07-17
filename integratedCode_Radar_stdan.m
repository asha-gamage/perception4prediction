%% Script Description
% This script takes in the Radar sensor readings along with the ego car global coordinates and calculates
% the global x,y coordinates of the traffic cars to be used to produce
% the .mat file to be fed into the CS-LSTM model to predict the trajectory of the target car

% 06/10/23:Updated the code to include the ground truth tracks (directly read from CM), such that the derivation of 'fut' which is the
% ground truth trajectory positions for the prediction horizon of 5 sec to the future to be used for the RMSE calculation is
% always available (as opposed to potentially missing detections if the sensor based future frames are used for the 'fut' calculation.

% 07/10/23: Updated the section which calculates the 'detectedcars' instead of using 'numTraffic' as per the original script updated
% on 29/03/23. Accordingly updated the initiation of 'measurements' cell array. Updated the .mat file getting saved to include
% 'tracksGndT' for calculating the 'fut' based on the ground-truth data for better calculations of RMSE for the prediction horizon.

% 20/10/23: Correct the 'for' loop syntax for correct number of iterations in thesection for generating the ‘traj’ data files for the CS-LSTM model.
% Modify the way how the empty cell array to log data related to sensor mesaurements, 'measuredData' is generated.
% Added a new section to both the MATLAB scripts to populate an excel file, 'No.of Detections' listing the number of detections of each traffic car by the sensor.

% 27/10/23: Added a 'Data Imputation' section to fill the missing detection timeframes with simple interpolated data to facilitate the correct operation of 
% the trajectory prediction model, STDAN

close all; % close all the figures
clear;
% This section calculates the equation of the line the nose of the vehicle is pointing at a given time.
Speed = 5;
Dir = 'Right';
Road = 'Straight';
Param = 'FoV';
Scenario = '_infront of Lead_';
Data_impute = false;
Spd = num2str(Speed);

% Navigate to the correct folder to 'read the results files from' and to 'store the output files'
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
filename = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\radarData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\No.of Detections.xlsx'];
names = {Param,"Target","Lead to Ego","Lead to Target","NIO", "Lexus"};
% writecell(names,filename);

% for a = 25:25:150 % a is the assigned sensor parameter value (Range)
for a = 30:30:180 % a is the assigned sensor parameter value (FoV)
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Radar\',Param,'\RelLonVel_',Spd,'kph\Scenario_',Road,'_LC_',Dir,Scenario,Param,'_',num2str(a),'.erg'];
    readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Radar\',Param,'\RelLonVel_',Spd,'kph\VerFoV@15deg\Scenario_',Road,'_LC_',Dir,'_HorFoV (',num2str(a),').erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Radar\',Param,'\RelLonVel_',Spd,'kph\Scenario_',Road,'_LC_',Dir,'_',Param,'_',num2str(a),'.erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Radar\',Param,'\RelLonVel_',Spd,'kph\VerFoV@15deg\Scenario_US101_LC_',Dir,'_infront of Lead_HorFoV (',num2str(a),').erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Radar\',Param,'\RelLonVel_',Spd,'kph\Scenario_',rd,'_',folder,'_2023_Range_',num2str(a),'.erg'];
%     readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\',rd,'\Manoeuvre_Cut in_',Dir,'\Radar\',Param,'\RelLonVel_',Spd,'kph\VerFoV@15deg\Scenario_',rd,'_',folder,'_2023_HorFoV (',num2str(a),').erg'];
   
    dFile = cmread(readPath);
    % Read the required data fields from the CM logged results file
    % Initial time-step removed to allow for radar model's cycle time
    ego_xFr1 = dFile.Car_Fr1_tx.data; % Ground truth x coordinate of car
    ego_yFr1 = dFile.Car_Fr1_ty.data; % Ground truth y coordinate of car
%     ego_vel = dFile.Car_v.data; % Longitudinal velocity of car
    ego_velX = dFile.Car_Fr1_vx.data; % X componant of velocity of the ego vehicle
    ego_velY = dFile.Car_Fr1_vy.data; % Y componant of velocity of the ego vehicle
    ego_acclX = dFile.Car_Fr1_ax.data; % X componant of acceleration of the ego vehicle   
    ego_acclY = dFile.Car_Fr1_ay.data; % X componant of acceleration of the ego vehicle   
    Time = dFile.Time.data; % Initial time-step removed to allow for radar model's cycle time
    rdRefAng = asin(dFile.RdVector_y.data(2:end));% custom created UAQ
    yawAng = dFile.Car_Yaw.data; % Ground truth yaw angle of ego car
    yawAng = round(yawAng*100)/100; % Convert the yawangle to 2 decimal point representation
    radarPos = 4.2; % Position of the radar sensor on the vehicle
    ego_x = ego_xFr1 + radarPos.*cos(yawAng); % Ground truth x coordinate of radar position
    ego_y = ego_yFr1 + radarPos.*sin(yawAng); % Ground truth y coordinate of radar position
    
    % Traffic car detection and ground truth data
    numTraffic = max(dFile.Sensor_Radar_Radar1_nObj.data); % Maximum number of objects detected by the radar sensor.
    %     The problem of using 'numTraffic' is that it captures the maximum number of objects detected at the instantaneous time frame, so getting the max of that vector could miss out on detected objects
    %     if the objects detected are varied but the total detected at a given time frame is less than the total of all detected objects. I.e. cars 1, 2, 3, 4 gets detected at different time frames during
    %     the total test run, however there's no time frame where all 4 are detected together.
    tt = struct2cell(dFile); % Convert the dFile structure to a cell, to enable reading the strings
    
    % This section extracts the required radar measurements and ground-truth data to subsequently calculate x, y coordinates
    % of the traffic cars
    detdCars = [];
    % Below 'for' loop get a list of all the traffic objects detected by the radar sensor
    for i = 1:numTraffic % numTraffic can be used here, since although the objectIDs(detected cars) could be more than 'numTraffic', it is the maximum objects detected at any given time step in the simulation duration
        traffic = ['Sensor.Radar.Radar1.Obj' num2str(i-1) '.ObjId'];
        % Loops through the entire data structure logged to match the Radar sensor detected global object ids assigned to each radar obj id
        for q = 1:length(tt)
            if convertCharsToStrings(tt{q}.name) == convertCharsToStrings(traffic)
                objIDs = unique(tt{q}.data);
                detdCars = [detdCars,objIDs]; % Generates a list of all the detected traffic objects
            end
        end
    end
    detdCars = detdCars(find(detdCars>=16000000)); % To filter only the traffic vehicles
    detdCars = unique(detdCars); % Removes duplication
    % measuredData = cell(length(detdCars),2);
    measuredData = cell(max(detdCars)-16000000+1,2); % Generate an empty cell array to log data related to sensor mesaurements
    measuredData(cellfun(@isempty,measuredData)) = {zeros(size(Time))}; % Pre-allocate memory to the empty cells, assigning zeros
    
    for i = 1:numTraffic
        rad_Idx = ['Sensor.Radar.Radar1.Obj' num2str(i-1)]; % Auto-generate the common string elements for reading data
        Trf_ID = [rad_Idx '.ObjId'];
        meas_X = [rad_Idx '.DistX'];
        meas_Y = [rad_Idx '.DistY'];
        meas_Dist = [rad_Idx '.Dist'];
        Detect = [rad_Idx '.MeasStat'];
        rel_Xvel = [rad_Idx '.VrelX'];
        rel_Yvel = [rad_Idx '.VrelY'];
        meas_Accel = [rad_Idx '.ArelX'];
%         course_Ang = [rad_Idx '.RelCourseAngle'];
        
        % Iterate through all the records to fill the cell with filtered data
        for q = 1:length(tt)
            % Fill the measuredData cell structure with the required data only.
            if convertCharsToStrings(tt{q}.name) == convertCharsToStrings(Trf_ID)
                objIDs = unique(tt{q}.data);
                objIDs = objIDs(find(objIDs>=16000000));
                % A loop is created to capture all the different traffic cars being captured under the same ObjId during the simulation run.
                % This happens due to the switching of object IDs in CM.
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
                        % Only consider data from the 2nd time frame to allow for missing data at initial
                        % time frame due to cycle time of the Radar model.
                        if convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_X)
                            measuredData{objIDs(obj)-16000000+1,1}(1,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_Y)
                            measuredData{objIDs(obj)-16000000+1,1}(2,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_Dist)
                            measuredData{objIDs(obj)-16000000+1,1}(3,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(Detect)
                            measuredData{objIDs(obj)-16000000+1,1}(4,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(rel_Xvel)
                            measuredData{objIDs(obj)-16000000+1,1}(5,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(rel_Yvel)
                            measuredData{objIDs(obj)-16000000+1,1}(6,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_Accel)
                            measuredData{objIDs(obj)-16000000+1,1}(7,tarIdx) = tt{k}.data(tarIdx);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_X)
                            measuredData{objIDs(obj)-16000000+1,2}(1,:) = tt{k}.data(1:end);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Y)
                            measuredData{objIDs(obj)-16000000+1,2}(2,:) = tt{k}.data(1:end);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Xvel)
                            measuredData{objIDs(obj)-16000000+1,2}(3,:) = tt{k}.data(1:end);  
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Yvel)
                            measuredData{objIDs(obj)-16000000+1,2}(4,:) = tt{k}.data(1:end);                             
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Xaccl)
                            measuredData{objIDs(obj)-16000000+1,2}(5,:) = tt{k}.data(1:end); 
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Yaccl)
                            measuredData{objIDs(obj)-16000000+1,2}(6,:) = tt{k}.data(1:end);
                        elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(laneID)
                            measuredData{objIDs(obj)-16000000+1,2}(7,:) = tt{k}.data(1:end);
                        end
                    end
                end
            end
        end
    end
    measuredData = cellfun(@(x) x(:,2:end), measuredData, 'UniformOutput', false); % Only consider data from the 2nd time frame to allow for missing data at initial
    % timeframe due to the 'cycle-time. of the Radar model.
    % Create a cell for the calculated x, y coordinates, velocity and acceleration for the traffic cars
    measurements = cell(length(measuredData),6);    
    
    % Calculate the x, y coordinates and X, y components of instant velocity and acceleration of the traffic cars using the radar readings
    %    for j = 1:length(detdCars)
    for j = (detdCars)-16000000+1
        %        for i = 1:length(dFile.Car_Yaw.data)-1 % -1 is to remove the initial data point which is not captured by the radar
        for i = 1:length(measuredData{j,1})%-1 % to ensure the loops iterates only up the number of detections by the radar and '-1' is to remove the initial data point which is not captured by the radar
            if  ~isempty(measuredData{j,1})% Check if the vehicle is detected by the sensor
                if measuredData{j,1}(4,i) == 3 % checks if detected by radar at a given time step
                    if yawAng(i) == 0 % if the ego vehicle trajectory is straight inline with the road center line (since road center line is set to the x-axis of the earth's fixed system).
                        measured_x = ego_x(i) + measuredData{j,1}(1,i);
                        measured_y = ego_y(i) + measuredData{j,1}(2,i); % assuming the measurements to the right of the line are negetive
                    else
                        theta = atan(measuredData{j,1}(2,i)/ measuredData{j,1}(1,i)); % Angle between the distance measurement line and the direction the vehicle is facing
                        Dist = sqrt(measuredData{j,1}(1,i)^2 + measuredData{j,1}(2,i)^2);
                        alpha = yawAng(i) + theta;
                        del_x = cos(alpha)* Dist;
                        del_y = sin(alpha)* Dist;
                        measured_x = ego_x(i) + del_x;
                        measured_y = ego_y(i) + del_y;
                    end
                    
                    VrelX = measuredData{j,1}(5,i);
                    VrelY = measuredData{j,1}(6,i);
%                     yawAngAbs = abs(yawAng(i));
                    
                    V_radX = VrelX*cos(yawAng(i)) + VrelY*sin(yawAng(i)) + ego_velX(i);% VrelY corrected from cos to sin()
                    V_radY = VrelY*cos(yawAng(i)) + VrelX*sin(yawAng(i)) + ego_velY(i);
            
                    ArelX = measuredData{j,1}(7,i);
                    A_radX = ArelX*cos(yawAng(i))+ ego_acclX(i);
                    A_radY = ArelX*sin(yawAng(i))+ ego_acclY(i);                    
                    
                else
                    measured_x = 0; measured_y = 0;
                    V_radX = 0; V_radY = 0;
                    A_radX = 0; A_radY = 0;               
                end
                
            else % added on 29/03/23
                measured_x = 0;   measured_y = 0;
                V_radX = 0; V_radY = 0;
                A_radX = 0; A_radY = 0;
            end
            measurements{j,1} = [measurements{j,1}, measured_x]; % Append the newly calculated x, y coordinates to the measurements cell structure
            measurements{j,2} = [measurements{j,2}, measured_y];
            measurements{j,3} = [measurements{j,3}, V_radX];
            measurements{j,4} = [measurements{j,4}, V_radY];
            measurements{j,5} = [measurements{j,5}, A_radX];
            measurements{j,6} = [measurements{j,6}, A_radY];
        end
        
        %% Data Imputation
        % This section fills the missing detection timeframes with simple interpolated data to facilitate the correct operation of the trajectory prediction model, STDAN 
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
                    senData(senData==0)=NaN; % replace the '0's with NaNs
                    xDataImputed = fillmissing(senData,'linear'); % Replace the missing detections, i.e. NaNs with linear interpolation
                    measurements{r,c}(non0start:non0end) = xDataImputed; % Replace the detected data range in the measurements vector with the filled solution
                end
            end
        end
        %% Differences between the ground-truths Vs the calculated equivalents based on radar measurements of the target car
        if  ~isempty(measuredData{j,1})
%             index = (measuredData{j,1}(4,:) == 3);
            index = (measurements{j,1} ~= 0); % get the indices of the imputed data
            %         index = index(2:end);
            calcCoordinate_x = measurements{j,1}(index);
            gTruth_x = measuredData{j,2}(1,:);
            gTruth_x_filtered = gTruth_x(index);
            diff_x = gTruth_x_filtered - calcCoordinate_x;
            
            compare_x = [gTruth_x_filtered', calcCoordinate_x', (gTruth_x_filtered - calcCoordinate_x)'];
            
            calcCoordinate_y = measurements{j,2}(index);
            gTruth_y = measuredData{j,2}(2,:);
            gTruth_y_filtered= gTruth_y(index);
            diff_y = gTruth_y_filtered - calcCoordinate_y;
            
            compare_y = [gTruth_y_filtered', calcCoordinate_y', (gTruth_y_filtered - calcCoordinate_y)'];
            
            % Compare the calculated Vs groundtruth X_velocity component
            index = (measurements{j,3} ~= 0); % get the indices of the imputed data
            calcVelX = measurements{j,3}(index);
            gTruth_VelX = measuredData{j,2}(3,:);
            gTruth_VelX_filtered = gTruth_VelX(index);
            diff_velX = gTruth_VelX_filtered - calcVelX;
            
            % Compare the calculated Vs groundtruth Y_velocity component
            calcVelY = measurements{j,4}(index);
            gTruth_VelY = measuredData{j,2}(4,:);
            gTruth_VelY_filtered = gTruth_VelY(index);
            diff_velY = gTruth_VelY_filtered - calcVelY;    
            
            % Compare the calculated Vs groundtruth X_acceleartion component
            index = (measurements{j,5} ~= 0); % get the indices of the imputed data
            calcAccX = measurements{j,5}(index);
            gTruth_AccX = measuredData{j,2}(5,:);
            gTruth_AccX_filtered = gTruth_AccX(index);
            diff_accX = gTruth_AccX_filtered - calcAccX;        
            
            % Compare the calculated Vs groundtruth Y_acceleartion component
            calcAccY = measurements{j,6}(index);
            gTruth_AccY = measuredData{j,2}(6,:);
            gTruth_AccY_filtered = gTruth_AccY(index);
            diff_accY = gTruth_AccY_filtered - calcAccY;   
            
            %% Plotting the X, Y co-ordinates from measurement-based Vs ground truths
            % close all;
            % Create a figure
            SurID = num2str(j-1);
            if SurID == '0'
                Title_1 = ['Ground Truths Vs Radar Measurements at ',Param,' ', num2str(a),'m for Target car',' ',rd];
            else
                Title_1 = ['Ground Truths Vs Radar Measurements at ',Param,' ', num2str(a),'m for Surround Car_', SurID,' ',rd];
            end
            figMetrics = figure('Name',Title_1);
            % This is the maximum figure width that can be used for publishing without clipping the subplots
            maxWidth = 1150;
            pos = figMetrics.Position;
            width = pos(3);
            figMetrics.Position = [pos(1)-(maxWidth-width)/2 pos(2) maxWidth pos(4)];
            
            % plot the radar measurements-based x coordinates
            xCord_radar = subplot(1,2,1);
            plot(calcCoordinate_x,'.');
            hold on
            plot(gTruth_x_filtered, '.');
            hold off            
            ylabel(xCord_radar,'x coordinate');
            xlabel(xCord_radar,'Point');
            title(xCord_radar,'Radar-based Vs Ground-truths for X coordinates');
            grid(xCord_radar,'on');
            legend ('Radar detections','GroundTruth','Location','northwest');
            
            % plot the radar measurements-based y coordinates
            yCord_radar = subplot(1,2,2);
            plot(calcCoordinate_y,'.');
            hold on
            plot(gTruth_y_filtered, '.');
            hold off
            ylabel(yCord_radar,'y coordinate');
            xlabel(yCord_radar,'Point');
            title(yCord_radar,'Radar-based Vs Groundtruths for Y coordinates');
            grid(yCord_radar,'on');
            legend ('Radar detections','GroundTruth','Location','northeast');
            
            % Define path for saving the figures
            FolderName = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\radarData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\graphs'];
            %[~, file]  = fileparts(filename);  % Remove extension
            saveas(gca, fullfile(FolderName, [Title_1, '.png']) ) % Append
            close all;
                                  
            % Plot the camera detection against the ground truth
            figure;
            if SurID == '0'
                Title_2 = ['Ground Truths Vs Radar based Detections at ',Param,' ', num2str(a),' for Target Car'];
            else
                Title_2 = ['Ground Truths Vs Radar based Detections at ',Param,' ', num2str(a),' for Surround Car_', SurID];
            end
            
            plot(calcCoordinate_x,calcCoordinate_y,'^','MarkerSize',1.5, 'LineWidth', 1.5);
            hold on
            plot(gTruth_x_filtered,gTruth_y_filtered,'-','LineWidth', 1);
            hold off
            ylabel('y coordinate');
            xlabel('x coordinate');
            if SurID == '0'
                title('Radar Detections Vs Ground-Truths for the Target Car ');
            else
                title(['Radar Detections Vs Ground-Truths for the Surround Car ', SurID]);
            end
            grid('on');
            legend ('Radar Detections','Ground-Truth','Location','northeast');
            
            % Define path for saving the figure
            FolderName = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\radarData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\graphs'];
            %[~, file]  = fileparts(filename);  % Remove extension
            saveas(gca, fullfile(FolderName, [Title_2, '.png']) ) % Append
            close all;
            
            %% Plotting the X, Y components of velocity from measurement-based Vs ground truths
%             % close all;
%             % Create a figure
%             SurID = num2str(j-1);
%             if SurID == '0'
%                 Title_3 = ['Ground Truths Velocity Vs Radar Measurements at ',Param,' ', num2str(a),'m for Target car',' ',rd];
%             else
%                 Title_3 = ['Ground Truths Velocity Vs Radar Measurements at ',Param,' ', num2str(a),'m for Surround Car_', SurID,' ',rd];
%             end
%             figMetrics = figure('Name',Title_3);
%             % This is the maximum figure width that can be used for publishing without clipping the subplots
%             maxWidth = 1150;
%             pos = figMetrics.Position;
%             width = pos(3);
%             figMetrics.Position = [pos(1)-(maxWidth-width)/2 pos(2) maxWidth pos(4)];
%             % plot the radar measurements-based velocity
%             velX_radar = subplot(1,2,1);
%             plot(calcVelX,'.');
%             hold on
%             plot(gTruth_VelX_filtered, '.');
%             hold off            
%             ylabel(velX_radar,'X velocity');
%             xlabel(velX_radar,'Point');
%             title(velX_radar,'Radar-based Vs Ground-truths for X-Velocity');
%             grid(velX_radar,'on');
%             legend ('Radar detections','GroundTruth','Location','northwest');
%             
%             % plot the radar measurements-based y coordinates
%             velY_radar = subplot(1,2,2);
%             plot(calcVelY,'.');
%             hold on
%             plot(gTruth_VelY_filtered, '.');
%             hold off
%             ylabel(velY_radar,'Y velocity');
%             xlabel(velY_radar,'Point');
%             title(velY_radar,'Radar-based Vs Groundtruths for Y-Velocity');
%             grid(velY_radar,'on');
%             legend ('Radar detections','GroundTruth','Location','northeast');

            % Plot the velocities
            figure;
            if SurID == '0'
                Title_3 = ['Ground Truths Vs Radar based Velocity at ',Param,' ', num2str(a),' for Target Car'];
            else
                Title_3 = ['Ground Truths Vs Radar based Velocity at ',Param,' ', num2str(a),' for Surround Car_', SurID];
            end
            V_rad = sqrt(calcVelX.^2+calcVelY.^2);
            V_GT = sqrt(gTruth_VelX_filtered.^2+gTruth_VelY_filtered.^2);

            plot(V_rad, '^','MarkerSize',1.5, 'LineWidth', 1.5);
            hold on
            plot(V_GT,'-','LineWidth', 1);
            hold off
            ylabel('Velocity');
            xlabel('point');
            if SurID == '0'
                title('Radar-based Vs Ground-Truths velocity for the Target Car ');
            else
                title(['Radar-based Vs Ground-Truths velocity for the Surround Car ', SurID]);
            end
            legend ('Radar Detections','Ground-Truth','Location','northeast');
            
            % Define path for saving the figure
            FolderName = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\radarData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\graphs'];
            %[~, file]  = fileparts(filename);  % Remove extension
            saveas(gca, fullfile(FolderName, [Title_3, '.png']) ) % Append
            close all;    
            
            % Plotting the X, Y components of acceleration from measurement-based Vs ground truths
            % close all;
            % Create a figure
            SurID = num2str(j-1);
            if SurID == '0'
                Title_4 = ['Ground Truths Acceleration Vs Radar Measurements at ',Param,' ', num2str(a),'m for Target car',' ',rd];
            else
                Title_4 = ['Ground Truths Acceleration Vs Radar Measurements at ',Param,' ', num2str(a),'m for Surround Car_', SurID,' ',rd];
            end
            figMetrics = figure('Name',Title_4);
            % This is the maximum figure width that can be used for publishing without clipping the subplots
            maxWidth = 1150;
            pos = figMetrics.Position;
            width = pos(3);
            figMetrics.Position = [pos(1)-(maxWidth-width)/2 pos(2) maxWidth pos(4)];
            % plot the radar measurements-based velocity
            accelX_radar = subplot(1,2,1);
            plot(calcAccX,'.');
            hold on
            plot(gTruth_AccX_filtered, '.');
            hold off            
            ylabel(accelX_radar,'X acceleration');
            xlabel(accelX_radar,'Point');
            title(accelX_radar,'Radar-based Vs Ground-truths for X-Acceleration');
            grid(accelX_radar,'on');
            legend ('Radar detections','GroundTruth','Location','northwest');
            
            % plot the radar measurements-based y coordinates
            accelY_radar = subplot(1,2,2);
            plot(calcAccY,'.');
            hold on
            plot(gTruth_AccY_filtered, '.');
            hold off
            ylabel(accelY_radar,'Y acceleration');
            xlabel(accelY_radar,'Point');
            title(accelY_radar,'Radar-based Vs Groundtruths for Y-Acceleration');
            grid(accelY_radar,'on');
            legend ('Radar detections','GroundTruth','Location','northeast');

            figure;
            if SurID == '0'
                Title_4 = ['Ground Truths Vs Radar based Acceleration at ',Param,' ', num2str(a),' for Target Car'];
            else
                Title_4 = ['Ground Truths Vs Radar based Acceleration at ',Param,' ', num2str(a),' for Surround Car_', SurID];
            end
            A_rad = sqrt(calcAccX.^2+calcAccY.^2);
            A_GT = sqrt(gTruth_AccX_filtered.^2+gTruth_AccY_filtered.^2);

            plot(A_rad, '^','MarkerSize',1.5, 'LineWidth', 1.5);
            hold on
            plot(A_GT,'-','LineWidth', 1);
            hold off
            ylabel('Acceleration');
            xlabel('point');
            if SurID == '0'
                title('Radar-based Vs Ground-Truths acceleration for the Target Car ');
            else
                title(['Radar-based Vs Ground-Truths acceleration for the Surround Car ', SurID]);
            end
            legend ('Radar Detections','Ground-Truth','Location','northeast');
            % Define path for saving the figure
            FolderName = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\radarData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\graphs'];
            %[~, file]  = fileparts(filename);  % Remove extension
            saveas(gca, fullfile(FolderName, [Title_4, '.png']) ) % Append
            close all;            
        end
    end
    
    %% Script for generating the traj data file for the STDAN model using CM generated data from the Radar measurements.
    
    % Read the data from the saved data file from CM
    surCars = [];
    %for t = 1:(detdCars)-16000000+1
    for t = (detdCars)-16000000+1
        if  ~isempty(measuredData{t,1})% Check if the vehicle is detected        
            trajTr = [10*Time(2:end)', measurements{t,2}', measurements{t,1}', measurements{t,3}', measurements{t,4}', measurements{t,5}', measurements{t,6}',measuredData{t,2}(7,:)', ones(length(Time)-1,1)*2]; % [time, local x, local y, v_VelX, v_VelY, v_AccX, v_AccY, lane id, v_type] correspond to the Target car (always given traffic name 'T00' in CM)
            vehID = ones(length(trajTr),1)*(t+15); % Generate the global vehicle IDs as in CM by multiplying the ones array by 16(In CM:16000000)
            trajTr = single([vehID, trajTr]); %  Form the first 8 fields of the matrix and convert to single precision
            [~,ia,~] = unique(measuredData{t,2}(1:3,2:end)','rows', 'stable'); % Remove the time frames matching the static positional data records from groundtruth data records           
            trajTr = trajTr(ia,:);
            if t==1 % This is the target car
                trajTar = [trajTr, rdRefAng(ia)']; % Concatenate the road reference angle
                non0start = find(trajTr(:,3) ~= 0, 1, 'first'); % index of first non-zero data recording
                non0end = find(trajTr(:,3) ~= 0, 1, 'last'); % index of first non-zero data recording
                trajtar = trajTr(non0start:non0end,:); % Time window the sensor had captured the target car's trajectory
            else
                surCars = [surCars;trajTr]; % For surround cars, impact of occlusions are accounted for since the 'no-data' points are not removed
            end
        end
    end
    
    % Create an array of all the trajectory data of the ego vehicle
    trajEgo = [ones(length(Time'),1)*15,10*Time', ego_yFr1', ego_xFr1', ego_velX', ego_velY', ego_acclX', ego_acclY', dFile.Car_Road_Lane_Act_LaneId.data',ones(length(Time'),1)*2]; % [vehID, time, local x, local y, v_VelX, v_VelY, v_AccX, v_AccY, landID, v_type]
    trajEgo = trajEgo(2:end,:);% Remove the first timeframe to align with the sensor-based dataset
    [~,ia,~] = unique(trajEgo(:,3:4),'rows', 'stable'); % Remove the static position data recording at the end of the scenario
    trajEgo = single(trajEgo(ia,:)); % Use 'ia' to index into and retrieve the rows that have unique combinations of elements in the 4th and 5th columns.
    % and convert the values into single precision.
    trajSur = [surCars;trajEgo];
    trajAll = [trajSur;trajtar];
    
    [traj, tracks] = CMinputs(trajTar, trajSur, trajAll);
    
    % Populate an Excel file to log the number of detections of traffic cars by the Radar sensor and a graph to plot the detections of
    % traffic during the simulation period.    
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
    
    title(['Traffic Detection by the Radar at ',Param,' ',num2str(a),unit]);
    Title_3 = ['\Traffic Detection by the Radar at ',Param,' ',num2str(a),unit];
    xlabel('Simulation Time');
%     ylim([0,5])
    ylim([0,6])
%     set(gca,'ytick', 0:1:5);
    set(gca,'ytick', 0:1:6);
    yticklabels({' ','Target car','Lead to Ego','Lead to Target', 'NIO', 'Lexus'});
%     yticklabels({' ','Target car','Lead to Ego','Lead to Target', 'NIO', 'Lexus'});
    %set(gca,'fontweight','bold')
%     xticks(0:50:300);
    xticks(0:50:600);
    ax.XAxis.MinorTick = 'on';
%     ax.XAxis.MinorTickValues = 0:10:300;
%     xticklabels({'0s','5s','10s', '15s','20s','25s','30s'})
    ax.XAxis.MinorTickValues = 0:10:600;
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
    %% Script for generating the traj data file for the STDAN model using CM based ground-truth data
    
    % Read the data from the saved data file from CM
    surCarsgT = [];
    for t = (detdCars)-16000000+1        
        trajTrGT = [10*Time(2:end)', measuredData{t,2}(2,:)', measuredData{t,2}(1,:)', measuredData{t,2}(3,:)', measuredData{t,2}(4,:)', measuredData{t,2}(5,:)', measuredData{t,2}(6,:)', measuredData{t,2}(7,:)', ones(length(Time)-1,1)*2]; % correspond to the Target car (always given traffic name 'T00' in CM)
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
    savePath = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\radarData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\matFiles\',rd,'_',Param,'_',num2str(a)];
    save(savePath,'traj','tracks','tracksGndT')
end

%% Script for generating the traj data file for the STDAN model using CM based ground-truth data
surCarsgT = [];
for t = (detdCars)-16000000+1         
    trajTrGT = [10*Time(2:end)', measuredData{t,2}(2,:)', measuredData{t,2}(1,:)', measuredData{t,2}(3,:)', measuredData{t,2}(4,:)', measuredData{t,2}(5,:)', measuredData{t,2}(6,:)', measuredData{t,2}(7,:)', ones(length(Time)-1,1)*2]; % correspond to the Target car (always given traffic name 'T00' in CM)
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
savePath = ['C:\Users\gamage_a\Documents\Python\stdan-master\',rd,'\radarData\manoeuvre_',folder,'\',Param,'\relVel_',Spd,'kph\matFiles\',rd,'_groundTruth'];
save(savePath,'traj','tracks','tracksGndT');

function [vel] = GTreads(xComp, yComp, ang)

vel = xComp.*cos(ang)-yComp.*sin(ang);

end

function [traj, tracks] = CMinputs(trajCM, trajTr, trajAll)

% Modified CS-LSTM and STDAN source code to determine the lateral and longitudinal
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
    if ub==ind || lb ==ind || trajCM(ub,4)==0 ||trajCM(lb,4)==0 % If current index is the start OR the end of the recording there's no longitudinal
        % reading (no measurement from the sensor) available....
        trajCM(k,13) = 1;% longitudinal maneuver is categorised as 'Normal speed' and adds it to a new column_8
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
            trajCM(k,13) = 1;% Const speed
        end
    end
    
    % Get grid locations:
    frameEgo = trajTr(and((trajTr(:,2) == time),(trajTr(:,9) == lane)),:);
    frameL = trajTr(and((trajTr(:,2) == time),(trajTr(:,9) == lane-1)),:);
    frameR = trajTr(and((trajTr(:,2) == time),(trajTr(:,9) == lane+1)),:);
    if ~isempty(frameL)
        for l = 1:size(frameL,1)
            y = frameL(l,4)-trajCM(k,4);
            if abs(y) < 27.432 % 90ft distance boundary
                gridInd = 1+round((y+27.432)/4.572);
                trajCM(k,13+gridInd) = frameL(l,1);
            end
        end
    end
    for l = 1:size(frameEgo,1)
        y = frameEgo(l,4)- trajCM(k,4);
        if abs(y) < 27.432 && y~=0
            gridInd = 14+round((y+27.432)/4.572);
            trajCM(k,13+gridInd) = frameEgo(l,1);
        end
    end
    if ~isempty(frameR)
        for l = 1:size(frameR,1)
            y = frameR(l,4)-trajCM(k,4);
            if abs(y) < 27.432
                gridInd = 27+round((y+27.432)/4.572);
                trajCM(k,13+gridInd) = frameR(l,1);
            end
        end
    end
end

tracksCM = {};
carIds = unique(trajAll(:,1)); % Create an array containing unique vehicleIDs
for l = 1:length(carIds)
    vehtrack = trajAll(trajAll(:,1) == carIds(l),2:10)'; % features and maneuver class id  ***Need to take the transpose for my research.
    % Iterate over the unique vehicleIDs and get the frameID, localX and localY and
    % transpose the matrix to a 3 by length of frame numbers captured
    tracksCM{1,carIds(l)} = vehtrack; % create a cell with references; DatasetID as the row and the vehicle ID as the column
end

% Remove time steps with no data for x, y coordinates
idx = find(trajCM(:,4));
trajCM = trajCM(idx,:);

%% Filter edge cases:
% Since the model uses 3 sec of trajectory history for prediction, the initial 3 seconds of each trajectory is not used for training/testing
% disp('Filtering edge cases...')
indsCM = zeros(size(trajCM,1),1);
for k = 1: size(trajCM,1)
    t = trajCM(k,2);
    if tracksCM{1,trajCM(k,1)}(1,31) <= t && tracksCM{1,trajCM(k,1)}(1,end)>t+1
        indsCM(k) = 1;
    end
end
trajCM = trajCM(find(indsCM),:);%  find(): Find indices of nonzero elements

% Add dataset Id column
traj = [ones(size(trajCM,1),1),trajCM];
tracks = tracksCM;
end



