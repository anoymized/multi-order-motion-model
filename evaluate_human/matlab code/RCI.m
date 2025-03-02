
%             tmpEPEGM=sqrt(( x(1,loc,mov)-GT(1,loc,mov))^2+( x(2,loc,mov)-GT(2,loc,mov))^2);
%             tmpEPERM=sqrt(( x(1,loc,mov)-Res(1,loc,mov))^2+( x(2,loc,mov)-Res(2,loc,mov))^2);
%             tmpEPERG=sqrt(( GT(1,loc,mov)-Res(1,loc,mov))^2+( GT(2,loc,mov)-Res(2,loc,mov))^2);
%             tmpEPEOG=sqrt(( GT(1,loc,mov))^2+( GT(2,loc,mov))^2);
%             tmpEPEOR=sqrt(( Res(1,loc,mov))^2+( Res(2,loc,mov))^2);
% 
% 
%             A(loc,mov,index)=tmpEPERG/(tmpEPEOG+tmpEPEOR); %GR/(OG+OR); 0~1
% 
%             a=Res(:,loc,mov)-GT(:,loc,mov); %GR
%             b=x(:,loc,mov)-GT(:,loc,mov); %GM
% 
% 
%             B(loc,mov,index)=(dot(a,b)/(sqrt(a(1)^2+a(2)^2)*sqrt(b(1)^2+b(2)^2))); %cos(GR,GM)
%             C(loc,mov,index)=tmpEPEGM/(tmpEPEGM+tmpEPERM); %GM/(GM+RM)
%             tmpRCI=(A(loc,mov,index)*B(loc,mov,index)*C(loc,mov,index));

          
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

%% predefined locations for each movie @2x (yx * 36 locations * 5 movies )
range=[150,275,1350,1475;350,475,1500,1625;450,575,800,925;275,400,1160,1285;570,695,1700,1825]; % y1,y2,x1,x2 @ 2x
Probexy=NaN(2,locN,movN);
for mov=1:movN
    Probexy(:,:,mov)=(combvec(range(mov,1):ProbeC2C:range(mov,2),range(mov,3):ProbeC2C:range(mov,4))); % y and x, 25 is
end


%% get response and GT(uv * 36 locations * 5 movies)
cd([pwd '/Data to  GT and human response'])

load('HumanResp.mat');
load('SintelGT.mat');
cd('../data to model')
flow1 = permute(load('movie01_layer0_7_8.mat').flow,[1,3,2]);
flow2 = permute(load('movie02_layer0_7_8.mat').flow,[1,3,2]);
flow3 = permute(load('movie03_layer0_7_8.mat').flow,[1,3,2]);
flow4 = permute(load('movie04_layer0_7_8.mat').flow,[1,3,2]);
flow5 = permute(load('movie05_layer0_7_8.mat').flow,[1,3,2]);



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


