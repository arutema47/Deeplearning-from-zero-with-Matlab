function [grad] = numerical_gradient(IN,W,t)
%NUMERICAL_GRADIENT この関数の概要をここに記述
%   詳細説明をここに記述
h= 1e-4;
grad=zeros(size(W));

for idx=1:size(W)
    for idx2=1:length(W)
        
        W1=W;
        W1(idx,idx2)=W(idx,idx2)+h;
        grad1=loss_func(IN,W1,t);
        
        W2=W;
        W2(idx,idx2)=W(idx,idx2)-h;
        
        grad2=loss_func(IN,W2,t);
        grad(idx,idx2)=(grad1-grad2)/(2*h);
        
    end
end

end

