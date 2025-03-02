clear
close
D=[355,5,15]; %Example
R = deg2rad( D ) ;
%% Circular  difference [Three methods get the same results]

% using atan
CirdiffR1=atan2(sin(R(1)-R(2)), cos(R(1)-R(2)));
CirdiffD=rad2deg(CirdiffR1);



R1D=circ_dist(R1,circ_mean(R1'));
R2D=circ_dist(R2,circ_mean(R2'));
num=sum(R1D .* R2D);
den=sqrt(sum(R1D.^2)*sum(R2D.^2));
circorr3 = num / den;

