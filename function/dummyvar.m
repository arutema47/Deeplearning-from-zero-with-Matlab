function [OUT] = dummyvar(IN)
%DUMMYVAR ���̊֐��̊T�v�������ɋL�q
%   �ڍא����������ɋL�q
OUT=zeros(length(IN):10);
for X=1:1:length(IN)
    for SEARCH=1:1:10
        if SEARCH==IN(X)
            OUT(X,SEARCH)=1;
        end
    end
end

end

