function [OUT] = sigmoid(IN)
%SIGMOID sigmoid function
OUT1=(1+exp(-IN));
OUT=OUT1.^(-1);
end

