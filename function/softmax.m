function [OUT] = softmax(IN)
%SOFTMAX
c=max(IN);
C1=IN-c;
exp_a=exp(C1);
sum_exp_a=sum(exp_a);
OUT=exp_a/sum_exp_a;



end

