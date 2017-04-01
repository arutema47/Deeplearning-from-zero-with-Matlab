function [OUT] = ReLU(IN)
%RELU RELU function
OUT=IN;
OUT(OUT<=0)=0; %max
end