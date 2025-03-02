clear
% Define directories and parameters
cvdir = '/home/4TSSD/';
datadir = '/home/szt/Research of KU/project_percived_flow';
datasetName = {'1_KITTI'}; % here only use KITTI 2015

cvname = {'dual-final'};
cvN = 1; sessionN = 12; movN = 2; flowN = 2; locN = 10; datasetN =1;
% Load the table
csvFile = fullfile(datadir, 'kittiperceivedflowtable.csv');
opts = detectImportOptions(csvFile);
opts.PreserveVariableNames = true;
HuPerFlow = readtable(csvFile, opts);

% Initialize output structure
CVflowmatrix = NaN(flowN, locN, movN, sessionN,datasetN, cvN);
Respflowmatrix = NaN(flowN, locN, movN, sessionN,datasetN);
GTflowmatrix = NaN(flowN, locN, movN, sessionN,datasetN);

% Main loop
for cv = 1:cvN
    for dataset = 1:datasetN
        for session = 1:sessionN
            for mov = 1:movN
                for loc = 1:locN
                    % Construct file path and load data
                    flowname = fullfile(cvdir, cvname{cv}, datasetName{dataset}, ...
                        ['session' sprintf('%03.0f', session)], ...
                        ['flow_MOV' num2str(mov) '_8-9.mat']);
                    if exist(flowname, 'file')
                        load(flowname)
                        % Filter rows in HuPerFlow
                        row = find(HuPerFlow.dataset == dataset & HuPerFlow.session == session & ...
                            HuPerFlow.movie == mov & HuPerFlow.location == loc);
                        if ~isempty(row)
                            % Extract coordinates and populate CVflowmatrix
                            tmpy = HuPerFlow.y_coordinate(row);
                            tmpx = HuPerFlow.x_coordinate(row);
                            if cv ==2
                                tmpuv = flow_vec(tmpy, tmpx, :);
                            else
                                tmpuv=flow(tmpy,tmpx,:);
                            end
                          
                            datasetidx = dataset;
                           
               
                            CVflowmatrix(1,loc,mov,session,datasetidx,cv)=tmpuv(1);
                            CVflowmatrix(2,loc,mov,session,datasetidx,cv)=-tmpuv(2); % this is critical, negative v vectors is up, opposite in polar
                            Respflowmatrix(:,loc,mov,session,datasetidx)=[HuPerFlow.Resp_u(row),HuPerFlow.Resp_v(row)];
                            GTflowmatrix(:,loc,mov,session,datasetidx)=[HuPerFlow.GT_u(row),HuPerFlow.GT_v(row)];
                        end
                    else
                        warning('File %s not found', flowname);
                    end
                end
            end
        end
    end
end

cd(cvdir)
save('AllCVflowmatrix.mat','CVflowmatrix');

for cv=1:cvN
    tmpcv=squeeze(CVflowmatrix(:,:,:,:,:,cv));
    summary(cv,1)=corr(tmpcv(:),GTflowmatrix(:));
    summary(cv,2)=corr(tmpcv(:),Respflowmatrix(:));
    summary(cv,3)=partialcorr(tmpcv(:),Respflowmatrix(:),GTflowmatrix(:));
    
end

