function [OUT,mask] = ReLU(IN)
%RELU RELU function
for X=1:size(IN,1)
for XX=1:size(IN,2)

if IN(X,XX)<=0
OUT(X,XX)=0;
mask(X,XX)=0;
else 
OUT(X,XX)=IN(X,XX);
mask(X,XX)=1;
end
end
end
end