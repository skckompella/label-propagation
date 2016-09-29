% this code generates 2D points for the two moon semi-supervised learning
% problem. Specify the number of labeled points below (ranging from a
% minimum of 1). 

load 2moons.mat;

l=1; % number of labeled examples

pos=find(y==1);
neg=find(y==-1);
ipos=randperm(length(pos));
ineg=randperm(length(neg));
y1=zeros(length(y),1);
y1(pos(ipos(1:l)))=1;
y1(neg(ineg(1:l)))=-1;

% generate a figure showing the points and the labels 

 figure;  plot2D(x,y1,12);