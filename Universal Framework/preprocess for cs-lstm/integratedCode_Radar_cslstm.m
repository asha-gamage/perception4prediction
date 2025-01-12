%% Script Description
% This script takes in the Radar sensor readings along with the ego car global coordinates and calculates
% the global x,y coordinates of the traffic cars to be used to produce
% the .mat file to be fed into the CS-LSTM model to predict the trajectory of the target car

% 06/10/23:Updated the code to include the ground truth tracks (directly read from CM), such that the derivation of 'fut' which is the
% ground truth trajectory positions for the prediction horizon of 5sec to the future to be used for the RMSE calculation is
% always available (as opposed to potentially missing detections if the sensor based future frames are used for the 'fut' calculation.

% 07/10/23: Updated the section which calculates the 'detectedcars' instead of using 'numTraffic' as per the original script updated
% on 29/03/23. Accordingly updated the initiation of 'measurements' cell array. Updated the .mat file getting saved to include
% 'tracksGndT' for calculating the 'fut' based on the ground-truth data for better calculations of RMSE for the prediction horizon.

% 20/10/23: Correct the 'for' loop syntax for correct number of iterations in thesection for generating the ‘traj’ data files for the CS-LSTM model.
% Modify the way how the empty cell array to log data related to sensor mesaurements, 'measuredData' is generated.
% Added a new section to both the MATLAB scripts to populate an excel file, 'No.of Detections' listing the number of detections of each traffic car by the sensor.

% 27/10/23: Added a 'Data Imputation' section to fill the missing detection timeframes with simple interpolated data to facilitate the correct operation of
% the trajectory prediction model, CS-LSTM

close all; % close all the figures
clear;

% Define possible values for each variable
Speeds = 5; %, 10, 15]; % Add more speeds if necessary
% Roads = {'US101', 'Straight', 'HW'}; %, 'Curve'};
Roads = {'HW'}; %, 'Curve'};
Dirs = {'Right', 'Left'};
Params = {'Range', 'FoV'};
% Scenarios = {'Infront of Ego', 'Infront of Lead'};
Scenarios = {''};
Data_imputes = {false, true};

% Iterate through all combinations of Speed, Dir, Road, and Param
for Speed = Speeds
    for Road = Roads
        for Dir = Dirs
            for Param = Params
                for Scenario = Scenarios
                    for Data_impute = Data_imputes
                        Spd = num2str(Speed);
                        
                        % Navigate to the correct folder based on the current values of Dir, Road, and Param
                        switch Dir{1}
                            case 'Right'
                                folder = 'RLC';
                            case 'Left'
                                folder = 'LLC';
                        end
                        
                        switch Road{1}
                            case 'Straight'
                                rd = 'StraightRd';             
                                names = {Param{1}, "Target", "Lead to Ego", "Lead to Target", "NIO", ""}; % Adjust if needed
                            case 'US101'
                                rd = 'US101_original';
                                names = {Param{1}, "Target", "Lead to Ego", "Lead to Target", "NIO", ""};
                            case 'HW'
                                rd = 'HW Freeway';                                
                                names = {Param{1},"Target","Lead to Ego","Lead to Target","NIO", "Lexus"}; % for HW Scenarios                                
                            case 'Curve'
                                rd = 'Road Curve';
                                
                        end
                        
                        switch Data_impute{1}
                            case false
                                if Road{1}=="HW"
                                    dFolder = 'Original';
                                else 
                                    dFolder = '_original';
                                end
                            case true
                                if Road{1}=="HW"
                                    dFolder = 'Data imputed';
                                else 
                                    dFolder = '_data imputed';
                                end
                        end
                        
                        switch Param{1}
                            case 'Range'
                                unit = 'm';
                                range_vals = 25:25:150; % Range values for 'Range' Param
                            case 'FoV'
                                unit = 'deg';
                                range_vals = 30:30:180; % FoV values for 'FoV' Param
                        end
                        
                        % Create an Excel file to record the number of detections
                        filename = ['C:\Users\gamage_a\Documents\Python\conv-social-pooling-master\', rd, '\radarData\manoeuvre_', folder, '\', Param{1}, '\relVel_', Spd, 'kph\No.of Detections.xlsx'];
                        writecell(names, filename);
                        
                        % Iterate through range values (for example, Range or FoV values)
                        for a = range_vals
                            if Param{1}=="Range"
                                if Road{1} =="US101"
                                    readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\', rd, '\Manoeuvre_Cut in_', Dir{1}, '\Radar\', Param{1}, '\RelLonVel_', Spd, 'kph\Scenario_', Road{1}, '_LC_', Dir{1}, '_',Scenario{1},'_', Param{1}, '_', num2str(a), '.erg'];
                                else
                                    readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\', rd, '\Manoeuvre_Cut in_', Dir{1}, '\Radar\', Param{1}, '\RelLonVel_', Spd, 'kph\Scenario_', Road{1}, '_LC_', Dir{1}, '_', Param{1}, '_', num2str(a), '.erg'];
                                end
                            elseif Param{1}=="FoV"
                                if Road{1} =="US101"
                                    readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\', rd, '\Manoeuvre_Cut in_', Dir{1}, '\Radar\', Param{1}, '\RelLonVel_', Spd, 'kph\VerFoV@15deg\Scenario_', Road{1}, '_LC_', Dir{1}, '_',Scenario{1},'_HorFoV (',num2str(a),').erg'];
                                else
                                    readPath = ['C:\Users\gamage_a\Documents\CM_Trail\SimOutput\WMGL241\', rd, '\Manoeuvre_Cut in_', Dir{1}, '\Radar\', Param{1}, '\RelLonVel_', Spd, 'kph\VerFoV@15deg\Scenario_', Road{1}, '_LC_', Dir{1}, '_HorFoV (',num2str(a),').erg'];
                                end
                            end
                            dFile = cmread(readPath);
                            % Read the required data fields from the CM logged results file
                            % Initial time-step removed to allow for radar model's cycle time
                            ego_xFr1 = dFile.Car_Fr1_tx.data; % Ground truth x coordinate of car
                            ego_yFr1 = dFile.Car_Fr1_ty.data; % Ground truth y coordinate of car
                            Time = dFile.Time.data; % Initial time-step removed to allow for radar model's cycle time
                            rdRefAng = asin(dFile.RdVector_y.data(2:end));% custom created UAQ
                            yawAng = dFile.Car_Yaw.data; % Ground truth yaw angle of ego car
                            yawAng = round(yawAng*100)/100; % Convert the yawangle to 2 decimal point representation
                            radarPos = 4.2; % Position of the radar sensor on the vehicleF
                            ego_x = ego_xFr1 + radarPos.*cos(yawAng); % Ground truth x coordinate of radar position
                            ego_y = ego_yFr1 + radarPos.*sin(yawAng); % Ground truth y coordinate of radar position
                            
                            
                            % Traffic car detection and ground truth data
                            numTraffic = max(dFile.Sensor_Radar_Radar1_nObj.data); % Maximum number of objects detected by the radar sensor.
                            %     The problem of using 'numTraffic' is that it captures the maximum number of objects detected at the instantaneous time frame, so getting the max of that vector could miss out on detected objects
                            %     if the objects detected are varied but the total detected at a given time frame is less than the total of all detected objects. I.e. cars 1, 2, 3, 4 gets detected at different time frames during
                            %     the total test run, however there's no time frame where all 4 are detected together.
                            
                            %     measuredData = cell(numTraffic,2); % Create a cell with 2 columns and a row for each object detected
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
                                        detdCars = [detdCars,objIDs]; % Generatse a list of all the detected traffic objects
                                    end
                                end
                            end
                            detdCars = detdCars(find(detdCars>=16000000)); % To filter only the traffic vehicles
                            detdCars = unique(detdCars); % Removes duplication
                            measuredData = cell(size(detdCars,2),2); % Generate an empty cell array to log data related to sensor mesaurements
                            measuredData(cellfun(@isempty,measuredData)) = {zeros(4,length(Time))}; % Pre-allocate memory to the empty cells, assigning zeros
                            
                            % Create a cell array matrix where the columns are the actual traffic vehicles in the scenario and the rows are the object IDs
                            % assigned to the detected cars by the sensor
                            TrfDet = {};
                            for i = 1:numTraffic % Here the loop is created assuming all traffic will be detected by the camera
                                Trf_ID = ['Sensor.Radar.Radar1.Obj' num2str(i-1) '.ObjId']; % Auto-generate the common string elements for reading data
                                for k = 1:length(tt)
                                    % Search for the elements capturing the objIDs and extract the indices of the time frame of detection of a vehicle and
                                    % log them in a cell array, 'TrfDet'
                                    if convertCharsToStrings(tt{k}.name) == convertCharsToStrings(Trf_ID)
                                        objIDs = unique(tt{k}.data);
                                        objIDs = objIDs(objIDs>=16000000);
                                        for detect = 1:length(objIDs)
                                            indexArray = tt{k}.data==(objIDs(detect));
                                            vehIndex = find(detdCars==(objIDs(detect)));
                                            TrfDet {i,vehIndex} = indexArray;
                                        end
                                    end
                                end
                            end
                            
                            [nObjIds,nVehIds] = size(TrfDet);% Obtain the number of ObjectIds and the number of vehicles
                            
                            % Log the ground truths for the detected traffic cars in the cell array - 'measuredData'
                            for d = 1:nVehIds
                                GT_traffic_X = ['Traffic.T0' num2str(d-1) '.tx'];
                                GT_traffic_Y = ['Traffic.T0' num2str(d-1) '.ty'];
                                laneID = ['Traffic.T0' num2str(d-1) '.Lane.Act.LaneId'];
                                
                                for k = 1:length(tt)
                                    if convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_X)
                                        measuredData{d,2}(1,:) = tt{k}.data;
                                    elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(GT_traffic_Y)
                                        measuredData{d,2}(2,:) = tt{k}.data;
                                    elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(laneID)
                                        measuredData{d,2}(3,:) = tt{k}.data;
                                    end
                                end
                            end
                            
                            % Log the radar detections of the traffic cars in the cell array - 'measuredData'
                            for i = 1:numTraffic % Here the loop is created based on the maximum traffic detected by the radar at any timeframe
                                rad_Idx = ['Sensor.Radar.Radar1.Obj' num2str(i-1)]; % Auto-generate the common string elements for reading data
                                Trf_ID = [rad_Idx '.ObjId'];
                                meas_X = [rad_Idx '.DistX'];
                                meas_Y = [rad_Idx '.DistY'];
                                meas_Dist = [rad_Idx '.Dist'];
                                Detect = [rad_Idx '.MeasStat'];
                                
                                % Iterate through all the records to extract the relevant filtered data to the cell array - 'measuredData'
                                for k = 1:length(tt)
                                    % Only consider data from the 2nd time frame to allow for missing data at initial
                                    % time frame due to cycle time of the Radar model.
                                    if convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_X)
                                        for vehIdx = 1:nVehIds
                                            if ~isempty(TrfDet{i,vehIdx})
                                                assignMatrix = measuredData{vehIdx,1}(1,:);
                                                assignMatrix(logical(TrfDet{i,vehIdx})) = tt{k}.data(logical(TrfDet{i,vehIdx}));
                                                measuredData{vehIdx,1}(1,:) = assignMatrix;
                                            end
                                        end
                                    elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_Y)
                                        for vehIdx = 1:nVehIds
                                            if ~isempty(TrfDet{i,vehIdx})
                                                assignMatrix = measuredData{vehIdx,1}(2,:);
                                                assignMatrix(logical(TrfDet{i,vehIdx})) = tt{k}.data(logical(TrfDet{i,vehIdx}));
                                                measuredData{vehIdx,1}(2,:) = assignMatrix;
                                            end
                                        end
                                    elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(meas_Dist)
                                        for vehIdx = 1:nVehIds
                                            if ~isempty(TrfDet{i,vehIdx})
                                                assignMatrix = measuredData{vehIdx,1}(3,:);
                                                assignMatrix(logical(TrfDet{i,vehIdx})) = tt{k}.data(logical(TrfDet{i,vehIdx}));
                                                measuredData{vehIdx,1}(3,:) = assignMatrix;
                                            end
                                        end
                                    elseif convertCharsToStrings(tt{k}.name) == convertCharsToStrings(Detect)
                                        for vehIdx = 1:nVehIds
                                            if ~isempty(TrfDet{i,vehIdx})
                                                assignMatrix = measuredData{vehIdx,1}(4,:);
                                                assignMatrix(logical(TrfDet{i,vehIdx})) = tt{k}.data(logical(TrfDet{i,vehIdx}));
                                                measuredData{vehIdx,1}(4,:) = assignMatrix;
                                            end
                                        end
                                    end
                                end
                            end
                            
                            measuredData = cellfun(@(x) x(:,2:end), measuredData, 'UniformOutput', false); % Only consider data from the 2nd time frame to allow for missing data at initial
                            % timeframe due to the 'cycle-time. of the Radar model.
                            % Create a cell for the calculated x, y coordinates for the traffic cars
                            measurements = cell(length(measuredData),2);
                            
                            % Calculate the x, y coordinates of the traffic cars using the radar readings
                            for j = 1:nVehIds
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
                                        else
                                            measured_x = 0;
                                            measured_y = 0;
                                        end
                                    else
                                        measured_x = 0;
                                        measured_y = 0;
                                    end
                                    measurements{j,1} = [measurements{j,1}, measured_x]; % Append the newly calculated x, y coordinates to the measurements cell structure
                                    measurements{j,2} = [measurements{j,2}, measured_y];
                                end
                                
                                %% Data Imputation
                                % This section fills the missing detection timeframes with simple interpolated data to facilitate the correct operation of the trajectory prediction model, CS-LSTM
                                if Data_impute{1}
                                    rows = length(measurements);
                                    for r = 1:rows
                                        for c = 1:2
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
                                
                                %% Differences between the ground-truth coordinates Vs the calculated coordinates based on radar measurements of the target car
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
                                    
                                    %% Plotting the X, Y co-ordinates from measurement-based Vs ground truths
                                    % close all;
                                    % Create a figure
                                    SurID = num2str(j-1);
                                    if SurID == '0'
                                        Title_1 = ['Ground Truths Vs Radar Measurements at ',Param{1},' ', num2str(a),'m for Target car',' ',rd];
                                    else
                                        Title_1 = ['Ground Truths Vs Radar Measurements at ',Param{1},' ', num2str(a),'m for Surround Car_', SurID,' ',rd];
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
                                    FolderName = ['C:\Users\gamage_a\Documents\Python\conv-social-pooling-master\',rd,'\radarData\manoeuvre_',folder,'\',Param{1},'\relVel_',Spd,'kph\graphs\', Scenario{1},dFolder];
                                    %[~, file]  = fileparts(filename);  % Remove extension
                                    saveas(gca, fullfile(FolderName, [Title_1, '.png']) ) % Append
                                    close all;
                                    
                                    % Plot the camera detection against the ground truth
                                    figure;
                                    if SurID == '0'
                                        Title_2 = ['Ground Truths Vs Radar based Detections at ',Param{1},' ', num2str(a),' for Target Car'];
                                    else
                                        Title_2 = ['Ground Truths Vs Radar based Detections at ',Param{1},' ', num2str(a),' for Surround Car_', SurID];
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
                                    FolderName = ['C:\Users\gamage_a\Documents\Python\conv-social-pooling-master\',rd,'\radarData\manoeuvre_',folder,'\',Param{1},'\relVel_',Spd,'kph\graphs\', Scenario{1},dFolder];
                                    %[~, file]  = fileparts(filename);  % Remove extension
                                    saveas(gca, fullfile(FolderName, [Title_2, '.png']) ) % Append
                                    close all;
                                    
                                end
                            end
                            
                            %% Script for generating the traj data file for the CS-LSTM model using CM generated data from the Radar measurements.
                            
                            % Read the data from the saved data file from CM
                            surCars = [];
                            for t = 1:nVehIds
                                if  ~isempty(measuredData{t,1})% Check if the vehicle is detected
                                    trajTr = [10*Time(2:end)', measurements{t,2}', measurements{t,1}', measuredData{t,2}(3,:)'];% [time, local x, local y, lane id] correspond to the Target car (always given traffic name 'T00' in CM)
                                    vehID = ones(length(trajTr),1)*(t+15); % Generate the global vehicle IDs as in CM by multiplying the ones array by 16(In CM:16000000)
                                    trajTr = single([vehID, trajTr]); %  Form the first 6 fields of the matrix and convert to single precision
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
                            trajEgo = [ones(length(Time'),1)*15,10*Time', ego_yFr1', ego_xFr1', dFile.Car_Road_Lane_Act_LaneId.data']; % [vehID, time, local x, local y, landID]
                            trajEgo = trajEgo(2:end,:);% Remove the first timeframe to align with the sensor-based dataset
                            [~,ia,~] = unique(trajEgo(:,3:4),'rows', 'stable'); % Remove the static position data recording at the end of the scenario
                            trajEgo = single(trajEgo(ia,:)); % Use 'ia' to index into and retrieve the rows that have unique combinations of elements in the 4th and 5th columns.
                            % and convert the values into single precision.
                            trajSur = [surCars;trajEgo];
                            trajAll = [trajSur;trajtar];
                            
                            [traj, tracks] = CMinputs(trajTar, trajSur, trajAll);
                            
                            % Populate an Excel file to log the number of detections of traffic cars by the Radar sensor and a grpah to plot the detections of
                            % traffic during the simulation period.
                            obsVehIDs = find(~cellfun(@isempty,tracks));
                            
                            figure;
                            ax = axes();
                            
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
                            
                            title(['Traffic Detection by the Radar at ',Param{1},' ',num2str(a),unit]);
                            Title_3 = ['\Traffic Detection by the Radar at ',Param{1},' ',num2str(a),unit];
                            xlabel('Simulation Time');
                            
                            if Road{1}=="HW"
                                ylim([0,6])
                                set(gca,'ytick', 0:1:6);
                                yticklabels({' ','Target car','Lead to Ego','Lead to Target', 'NIO', 'Lexus'});
                                xticks(0:100:800);
                                ax.XAxis.MinorTick = 'on';
                                ax.XAxis.MinorTickValues = 0:10:800;
                                xticklabels({'0s','10s', '20s','30s','40s', '50s','60s','70s','80s'})
                            else
                                ylim([0,5])
                                set(gca,'ytick', 0:1:5);
                                yticklabels({' ','Target car','Lead to Ego','Lead to Target', 'NIO'});
                                %     set(gca,'fontweight','bold')
                                xticks(0:50:800);
                                ax.XAxis.MinorTick = 'on';
                                ax.XAxis.MinorTickValues = 0:10:800;
                                xticklabels({'0s','5s','10s', '15s','20s','25s','30s','35s','40s', '45s','50s','55s','60s'})                                
                            end
                            
                            lcDet = traj(:,8);
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
                            
                            writecell(D,filename,'WriteMode','append');
                            %% Script for generating the traj data file for the CS pooling model using CM based ground-truth data
                            
                            % Read the data from the saved data file from CM
                            surCarsgT = [];
                            for t = 1:nVehIds
                                trajTrGT = [10*Time(2:end)', measuredData{t,2}(2,:)', measuredData{t,2}(1,:)', measuredData{t,2}(3,:)'];% correspond to the Target car (always given traffic name 'T00' in CM)
                                vehID = ones(length(trajTrGT),1)*(t+15); % Generate the global vehicle IDs as in CM by multiplying the ones array by 16(In CM:16000000)
                                trajTrGT = single([vehID, trajTrGT]); %  Format the first 6 fields of the matrix and convert to single precision
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
                            savePath = ['C:\Users\gamage_a\Documents\Python\conv-social-pooling-master\',rd,'\radarData\manoeuvre_',folder,'\',Param{1},'\relVel_',Spd,'kph\matFiles\', Scenario{1},dFolder,'\',rd,'_',Param{1},'_',num2str(a)];
                            save(savePath,'traj','tracks','tracksGndT')
                            
                        end
                        %% Script for generating the traj data file for the CS-LSTM model using CM based ground-truth data
                        surCarsGT = [];
                        for t = 1:nVehIds
                            trajTrGT = [10*Time(2:end)', measuredData{t,2}(2,:)', measuredData{t,2}(1,:)', measuredData{t,2}(3,:)'];% correspond to the Target car (always given traffic name 'T00' in CM)
                            vehID = ones(length(trajTrGT),1)*(t+15); % Generate the global vehicle IDs as in CM by multiplying the ones array by 16(In CM:16000000)
                            trajTrGT = single([vehID, trajTrGT]); %  Format the first 6 fields of the matrix and convert to single precision
                            [~,ia,~] = unique(trajTrGT(:,3:4),'rows', 'stable'); % Remove the static poisition data recording at the end of the scenario
                            trajTrGT = trajTrGT(ia,:);
                            if t==1
                                trajTarGT = [trajTrGT, rdRefAng(ia)'];
                            else
                                surCarsGT = [surCarsGT;trajTrGT];
                            end
                        end
                        
                        trajSurGT = [surCarsGT; trajEgo]; % Combine the surround vehicles together
                        trajAllGT = [trajSurGT; trajTarGT(:,1:end-1)];
                        
                        [traj, tracks] = CMinputs(trajTarGT, trajSurGT, trajAllGT);
                        
                        %% Save mat files:
                        %disp('Saving mat files...')
                        savePath = ['C:\Users\gamage_a\Documents\Python\conv-social-pooling-master\',rd,'\radarData\manoeuvre_',folder,'\',Param{1},'\relVel_',Spd,'kph\matFiles\', Scenario{1},dFolder,'\',rd,'_groundTruth'];
                        save(savePath,'traj','tracks','tracksGndT');
                    end
                    
                end
            end
        end
    end
end

function [traj, tracks] = CMinputs(trajCM, trajTr, trajAll)

% Modified N.Deo's source code to determine the lateral and longitudinal
% maneuvers: Same approach followed as implemented when using the NGSIM data.
for k = 1:length(trajCM(:,1))
    time = trajCM(k,2);
    lane = trajCM(k,5);
    ind = find(trajCM(:,2)==time); % Find the index of the vehicle trajectory where the frame ID (column 3) matches the 'time'.I.e. find the index of
    % the exact track from the list of tracks for a vehicle: vehid
    % find(): Find indices of nonzero elements
    
    % Get lateral maneuver:
    ub = min(size(trajCM,1),ind+40);% Upper bound is calculated by checking whether the index at each record has 40
    %(0.5*8sec duration used for observation and prediction) more Frame_IDs to the future and taking the lowest timeFrame
    % size(X,1) returns the number of rows of X and size(X,[1 2]) returns a row vector containing the number of rows & columns.
    lb = max(1, ind-40);% Lower bound is calculated by checking whether the index at each record has 40 more Frame_IDs to the past and taking
    % the highest Frame_ID
    if trajCM(ub,5)>trajCM(ind,5) || trajCM(ind,5)>trajCM(lb,5)% future lane Id > current lane Id OR current lane Id > past lane Id
        trajCM(k,7) = 2;% Categorise as 'Right Lane-change' and adds it to a new column_7
    elseif trajCM(ub,5)<trajCM(ind,5) || trajCM(ind,5)<trajCM(lb,5)
        trajCM(k,7) = 3;% Left Lane-change
    else
        trajCM(k,7) = 1;% Lane Keep
    end
    
    % Get longitudinal maneuver:
    ub = min(size(trajCM,1),ind+50);% Upper bound is calculated by checking whether the index at each record has 50 more frames(5s to the future)to
    % the future and taking the lowest duration
    lb = max(1, ind-30);% Lower bound is calculated by checking whether the index at each record has 30 frames to the past or checking if at the start
    % of the recording
    if ub==ind || lb ==ind || trajCM(ub,4)==0 ||trajCM(lb,4)==0 % If current index is the start OR the end of the recording there's no longitudinal
        % reading (no measurement from the sensor) available....
        trajCM(k,8) = 1;% longitudinal maneuver is categorised as 'Normal speed' and adds it to a new column_8
    else
        vHist = (trajCM(ind,4)-trajCM(lb,4))/(ind-lb);% Historical velocity calculated by dividing the longitudinal distance between
        % current and lower bound time frames
        vFut = (trajCM(ub,4)-trajCM(ind,4))/(ub-ind);% Future velocity calculated by dividing the longitudinal distance between
        % current and lower bound time frames
        if vFut/vHist < 0.8% vehicle to be performing a braking maneuver if it’s average speed over the prediction horizon is less
            % than 0.8 times its speed at the time of the prediction
            trajCM(k,8) = 2;% Braking and adds it to a new column_8
        else
            trajCM(k,8) = 1;
        end
    end
    
    % Get grid locations:
    frameEgo = trajTr(and((trajTr(:,2) == time),(trajTr(:,5) == lane)),:);
    frameL = trajTr(and((trajTr(:,2) == time),(trajTr(:,5) == lane-1)),:);
    frameR = trajTr(and((trajTr(:,2) == time),(trajTr(:,5) == lane+1)),:);
    if ~isempty(frameL)
        for l = 1:size(frameL,1)
            y = frameL(l,4)-trajCM(k,4);
            if abs(y) < 27.432 % 90ft distance boundary
                gridInd = 1+round((y+27.432)/4.572); % Filling the first column of the 13x3 spatial grid
                trajCM(k,8+gridInd) = frameL(l,1);
            end
        end
    end
    for l = 1:size(frameEgo,1)
        y = frameEgo(l,4)- trajCM(k,4);
        if abs(y) < 27.432 && y~=0
            gridInd = 14+round((y+27.432)/4.572);% 14 come from 90ft/15ft to the front and back of the ego car +1 (13) allocated to the frameL earlier.
            trajCM(k,8+gridInd) = frameEgo(l,1);
        end
    end
    if ~isempty(frameR)
        for l = 1:size(frameR,1)
            y = frameR(l,4)-trajCM(k,4);
            if abs(y) < 27.432
                gridInd = 27+round((y+27.432)/4.572);% 27 comes from (14 + 13) as above.
                trajCM(k,8+gridInd) = frameR(l,1);
            end
        end
    end
end

tracksCM = {};
%trajSet = trajTr(trajTr(:,1)==k,:); % I would question at this stage, why trajAll was created at the first place? Code would have been much
% simpler without. Here, it is seperating the Training trajectory set by the Dataset ID
carIds = unique(trajAll(:,1)); % Create an array containing unique vehicleIDs
for l = 1:length(carIds)
    vehtrack = trajAll(trajAll(:,1) == carIds(l),2:4)'; % ***Need to take the transpose for my research.
    % Iterate over the unique vehicleIDs and get the frameID, localX and localY and
    % transpose the matrix to a 3 by length of frame numbers captured
    tracksCM{1,carIds(l)} = vehtrack; % create a cell with references; DatasetID as the row and the vehicle ID as the column
end

% % Remove only the leading and lagging zeros to the sensor captured data for the target car
% non0start = find(trajCM(:,3) ~= 0, 1, 'first'); % index of first non-zero data recording
% non0end = find(trajCM(:,3) ~= 0, 1, 'last'); % index of first non-zero data recording
% trajCM = trajCM(non0start:non0end,:); % Limit the predictions on the time window the sensor had captured the target car's trajectory including occlusions

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



