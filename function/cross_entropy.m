function [OUT] = cross_entropy(IN,t)
%CROSS_ENTROPY 

batch_size=size(IN,1);

delta=1e-7;
OUT1=log(IN+delta);
OUT=-dot(t,OUT1); %sum and multi
OUT=sum(OUT)/batch_size;
end

