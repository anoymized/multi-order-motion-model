clear
close
clc
%% Sintel info
MovieName={'ambush_2','ambush_4','cave_4', 'market_6','temple_3'};
MovieName_={'mv1','mv2','mv3', 'mv4','mv5'};
MovieFrame=[110,95,126,105,115]; % Probed frame (Total Frames: 421,421,379,379,295)
picsize=[436,1024]*2;%@2x
uvN=2;
movN=5;
locN=36;
TrialN=movN*locN;
NeuronN=2;
ProbeC2C=25;%pixels
data_format_1 = 'movie%02d_layer%d_7_8.mat';
data_format_2 = 'movie%02d_layer%d_7_8.mat';
layer = 0;

range=[150,275,1350,1475;350,475,1500,1625;450,575,800,925;275,400,1160,1285;570,695,1700,1825]; % y1,y2,x1,x2 @ 2x
Probexy=NaN(2,locN,movN);
for mov=1:movN
    Probexy(:,:,mov)=(combvec(range(mov,1):ProbeC2C:range(mov,2),range(mov,3):ProbeC2C:range(mov,4))); % y and x, 25 is
end

Probexy=Probexy*2; % now become 4K

%% get response and GT(uv * 36 locations * 5 movies)
cd([pwd '/Data'])
load('HumanResp.mat');
load('SintelGT.mat');

cd('../path to model response/')
   for mov = 1:movN
        filelist{mov}=sprintf(data_format_1,[mov,layer]);
    end
    flow1_1 = permute(load(filelist{1}).flow,[1,3,2]);
    flow2_1 = permute(load(filelist{2}).flow,[1,3,2]);
    flow3_1 = permute(load(filelist{3}).flow,[1,3,2]);
    flow4_1 = permute(load(filelist{4}).flow,[1,3,2]);
    flow5_1 = permute(load(filelist{5}).flow,[1,3,2]);


    for mov = 1:movN
        filelist{mov}=sprintf(data_format_2,[mov,layer]);
    end
    flow1_2 = permute(load(filelist{1}).flow,[1,3,2]);
    flow2_2 = permute(load(filelist{2}).flow,[1,3,2]);
    flow3_2 = permute(load(filelist{3}).flow,[1,3,2]);
    flow4_2 = permute(load(filelist{4}).flow,[1,3,2]);
    flow5_2 = permute(load(filelist{5}).flow,[1,3,2]);


    flow1 = (flow1_2+flow1_2)./2;
    flow2 = (flow2_2+flow2_2)./2;
    flow3 = (flow3_2+flow3_2)./2;
    flow4 = (flow4_2+flow4_2)./2;
    flow5 = (flow5_2+flow5_2)./2;


flow1=imresize(flow1,2);
flow2=imresize(flow2,2);
flow3=imresize(flow3,2);
flow4=imresize(flow4,2);
flow5=imresize(flow5,2);
%% this is for 4K images
    data1_x =  diag(flow1(Probexy(2,:,1),Probexy(1,:,1),1));
    data1_y =  diag(flow1(Probexy(2,:,1),Probexy(1,:,1),2));
    
    data2_x =  diag(flow2(Probexy(2,:,2),Probexy(1,:,2),1));
    data2_y =  diag(flow2(Probexy(2,:,2),Probexy(1,:,2),2));
    
    data3_x =  diag(flow3(Probexy(2,:,3),Probexy(1,:,3),1));
    data3_y =  diag(flow3(Probexy(2,:,3),Probexy(1,:,3),2));
    
    data4_x =  diag(flow4(Probexy(2,:,4),Probexy(1,:,4),1));
    data4_y =  diag(flow4(Probexy(2,:,4),Probexy(1,:,4),2));
    
    data5_x =  diag(flow5(Probexy(2,:,5),Probexy(1,:,5),1));
    data5_y =  diag(flow5(Probexy(2,:,5),Probexy(1,:,5),2));


    X = [data1_x;data2_x;data3_x;data4_x;data5_x; data1_y;data2_y;data3_y;...
        data4_y;data5_y]';
    temp = [[data1_x,data1_y];[data2_x,data2_y];[data3_x,data3_y];...
        [data4_x,data4_y];[data5_x,data5_y]];
    [X_theta,X_speed ]= cart2pol(temp(:,1)',temp(:,2)');
    y=reshape(HumanResp,uvN,TrialN); % become 2*180, or using SintelGT
    GT=reshape(SintelGT,uvN,TrialN);

     % EPE error
    data_u =  [data1_x;data2_x;data3_x;data4_x;data5_x]';
    data_v = [data1_y;data2_y;data3_y;...
    data4_y;data5_y]';
    human_u = y(1,:);
    human_v = y(2,:);
    GT_u = GT(1,:);
    GT_v = GT(2,:);

    EPE_model_human = mean(sqrt((data_u - human_u).^2 + (data_v - human_v).^2))
    EPE_model_GT = mean(sqrt((data_u - GT_u).^2 + (data_v - GT_v).^2))

    [GT_theta,GT_speed ]= cart2pol(GT(1,:),GT(2,:));
    [Y_theta,Y_speed ]= cart2pol(y(1,:),y(2,:));
    y = [y(1,:),y(2,:)];
    GT = [GT(1,:),GT(2,:)];
    % 
    % X_theta = circ_dist(X_theta, circ_mean(X_theta'));
    % Y_theta = circ_dist(Y_theta ,circ_mean(Y_theta'));
 
    X_theta=mod(X_theta+2*pi,2*pi);
    GT_theta=mod(GT_theta+2*pi,2*pi);
    Y_theta=mod(Y_theta+2*pi,2*pi);


    Theta_diff = (X_theta - GT_theta);
    X_theta = X_theta - 2*pi*(Theta_diff>pi) +2*pi*(Theta_diff< -pi);
    Theta_diff = (Y_theta - GT_theta);
    Y_theta = Y_theta - 2*pi*(Theta_diff>pi) +2*pi*(Theta_diff< -pi);

    
    
    Pre = [ones(360,1),X'];
    XGP = Pre\y';
    
    Xpred = X'*XGP(2) + XGP(1);
    

  
    SSTX = var(y)*359;
    SSEX = (Xpred' - y)*(Xpred' - y)';
    R2_uv = 1-SSEX/SSTX
    corrHuman_uv = corr(X',y')
    corrHuman_speed = corr(X_speed',Y_speed')
    corrHuman_theta = corr(X_theta',Y_theta')

    corrGT_uv = corr(X',GT')
    corrGT_speed = corr(X_speed',GT_speed')
    corrGT_theta = corr(X_theta',GT_theta')
    
    
    
 
    Pre = [ones(180,1),X_speed'];
    XGP = Pre\Y_speed';
    
    X_speedpred = X_speed'*XGP(2) + XGP(1);
    
    Pre = [ones(180,1),X_theta'];
    XGP = Pre\Y_theta';
    
    X_thetapred = X_theta'*XGP(2) + XGP(1);
    
    
    SST_theta = var(Y_theta)*179;
    SST_speed = var(Y_speed)*179;
    
    SSE_theta = (X_thetapred' - Y_theta)*(X_thetapred' - Y_theta)';
    R2_theta = 1-SSE_theta/SST_theta
    
    SSE_speed = (X_speedpred' - Y_speed)*(X_speedpred' - Y_speed)';
    R2_speed = 1-SSE_speed/SST_speed
    
    
    pcor_uv = partialcorr(X',y',GT')
    pcor_speed = partialcorr(X_speed',Y_speed',GT_speed')
    pcor_theta = partialcorr(X_theta',Y_theta',GT_theta')
 
    uv_corr_list(layer+1,:) = corrHuman_uv;

       
    speed_corr_list(layer+1,:) = corrHuman_speed;
    theta_corr_list(layer+1,:) = corrHuman_theta;
    
    uv_r2_list (layer+1,:) = R2_uv;
    speed_r2_list(layer+1,:) = R2_speed;
    theta_r2_list(layer+1,:) = R2_theta;
    
    uv_pcor_list(layer+1,:) = pcor_uv;
    speed_pcor_list (layer+1,:) = pcor_speed;
    theta_pcor_list (layer+1,:) = pcor_theta;

    EPE_model_gt_list(layer+1,:) = EPE_model_GT;
    EPE_model_human_list(layer+1,:) = EPE_model_human;

    

    
    X = [data1_x,data1_y; data2_x,data2_y;data3_x,data3_y;data4_x,data4_y;data5_x,data5_y]';
    Y=reshape(HumanResp,uvN,TrialN); % become 2*180, or using SintelGT
    GT=reshape(SintelGT,uvN,TrialN);
    
    % y = [y(1,:),y(2,:)];
    % GT = [GT(1,:),GT(2,:)];
    
    
    
    
    tmpEPEGM = sqrt((X(1,:) - GT(1,:)).^2+(X(2,:) - GT(2,:)).^2);
    
    tmpEPERM =  sqrt((X(1,:) - Y(1,:)).^2+(X(2,:) - Y(2,:)).^2);
    tmpEPERG = sqrt((GT(1,:) - Y(1,:)).^2+(GT(2,:) - Y(2,:)).^2);
    
    tmpEPEOG= sqrt(GT(1,:) .^2+GT(2,:) .^2);
    
    tmpEPEOR= sqrt(Y(1,:).^2+Y(2,:) .^2);
    A = tmpEPERG./(tmpEPEOG+tmpEPEOR);
    a = Y -GT;
    b = X -GT;
    B = (dot(a,b)./(sqrt(a(1,:).^2+a(2,:).^2).*sqrt(b(1,:).^2+b(2,:).^2)));
    C = tmpEPEGM./(tmpEPEGM+tmpEPERM);
    tmpRCI = mean(A.*B .*C)

    RCI_list(layer+1,:) = tmpRCI

%% 
[X_theta,X_speed ]= cart2pol(temp(:,1)',temp(:,2)');
y=reshape(HumanResp,uvN,TrialN); % become 2*180, or using SintelGT
GT=reshape(SintelGT,uvN,TrialN);
[GT_theta,GT_speed ]= cart2pol(GT(1,:),GT(2,:));
[Y_theta,Y_speed ]= cart2pol(y(1,:),y(2,:));

human_u = y(1,:);
human_v = y(2,:);
GT_u = GT(1,:);
GT_v = GT(2,:);

y = [y(1,:),y(2,:)];
GT = [GT(1,:),GT(2,:)];
%
% X_theta = circ_dist(X_theta, circ_mean(X_theta'));
% Y_theta = circ_dist(Y_theta ,circ_mean(Y_theta'));
% Theta_diff = abs(X_theta - Y_theta);
% X_theta = X_theta + 2*pi*(Theta_diff>pi & X_theta<0) -2*pi*(Theta_diff>pi & X_theta>0);

%% theta would be -pi~pi, transfer to 0~2pi
X_theta=mod(X_theta+2*pi,2*pi);
GT_theta=mod(GT_theta+2*pi,2*pi);
Y_theta=mod(Y_theta+2*pi,2*pi);

X = [data1_x;data2_x;data3_x;data4_x;data5_x; data1_y;data2_y;data3_y;...
    data4_y;data5_y]';
%% get R^2
% ymean=mean(y(:));
% SStotal=sum((y(:)-ymean).^2);
% SSres=sum((y(:)-PredictData(:)).^2);
% R2=1-(SSres/SStotal);
%%
SSTX = var(y)*359;
SSEX = (Xpred' - y)*(Xpred' - y)';
R2X = 1-SSEX/SSTX
disp("pearson corrlation in (u,v) space")

corr(X',y')

%%



%%
Pre = [ones(180,1),X_speed'];
XGP = Pre\Y_speed';

X_speedpred = X_speed'*XGP(2) + XGP(1);

Pre = [ones(180,1),X_theta'];
XGP = Pre\Y_theta';

X_thetapred = X_theta'*XGP(2) + XGP(1);


SST_theta = var(Y_theta)*179;
SST_speed = var(Y_speed)*179;

SSE_theta = (X_thetapred' - Y_theta)*(X_thetapred' - Y_theta)';
R2_theta = 1-SSE_theta/SST_theta

SSE_speed = (X_speedpred' - Y_speed)*(X_speedpred' - Y_speed)';
R2_speed = 1-SSE_speed/SST_speed


figure()
subplot(1,3,1)
scatter(X,y,40,[0.1 0.1 0.74],"filled","MarkerFaceAlpha",0.5);hold on
plot(X,Xpred,'LineWidth',2);
legend(["Human","prediction"])
title("(u,v)")
grid on

subplot(1,3,2)
plot(X_theta,Y_theta,'.'); hold on;
plot(X_theta,X_thetapred,'LineWidth',2);
legend(["Human","prediction"])
title("Theta")
grid on

subplot(1,3,3)
plot(X_speed,Y_speed,'.'); hold on;
plot(X_speed,X_speedpred,'LineWidth',2);
legend(["Human","prediction"])
title("Speed")
grid on

disp("pearson corrlation: speed")
corr(X_speed',Y_speed')

disp("pearson corrlation: direction")
corr(X_theta',Y_theta')


figure()
polarscatter(Y_theta,Y_speed,100,[0.1 0.2 0.74],"filled","MarkerFaceAlpha",0.5)
hold on
polarscatter(X_theta,X_speed,100,[0.74 0.2 0.1],"filled","MarkerFaceAlpha",0.5)

legend(["human","model"])

partial_corr = partialcorr(X',y',GT');
partial_corr
% Define the font size and typeface
set(gca, 'FontName', 'Times New Roman');
% Define the line properties
set(gca, 'LineWidth', 2);
% Define the axis properties
set(gca, 'TickDir', 'out');
set(gca, 'Box', 'off');
% Define the legend properties
legend('Location', 'best');
set(gca, 'FontWeight', 'bold', 'FontSize', 20, 'FontName', 'Times New Roman', 'rTickLabel', get(gca, 'rTickLabel'));
%%
