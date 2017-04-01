function [outW] = pruning(W,threshold)
%PRUNING この関数の概要をここに記述
%   prune weights below threshold
%   W.W..weight
%   W.mask..0の位置
outW.mask=zeros(size(W));
outW.W=W;
for X=1:1:size(W,1)
    for XX=1:1:size(W,2)
        if (abs(W(X,XX))<threshold)
            outW.W(X,XX)=0; %prune
        else
            outW.mask(X,XX)=1;
        end
    end
end
            




end

