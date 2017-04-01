function [OUT] = dummyvar(IN)
%DUMMYVAR この関数の概要をここに記述
%   詳細説明をここに記述
OUT=zeros(length(IN):10);
for X=1:1:length(IN)
    for SEARCH=1:1:10
        if SEARCH==IN(X)
            OUT(X,SEARCH)=1;
        end
    end
end

end

