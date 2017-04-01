function [W_k,W_average] = w_quant_init(W,k,loop)
%WEIGTH_QUANT ‚±‚ÌŠÖ”‚ÌŠT—v‚ð‚±‚±‚É‹Lq
%   k-means

W_k=round((k-1)*rand(size(W)))+1; %k-rand


for COUNT=1:1:loop

%% average of each k
W_sum=zeros(k);
W_count=zeros(k);
for K=1:1:k
    for X=1:1:size(W_k,1)
        for XX=1:1:size(W_k,2)
            if W_k(X,XX)==K
                W_sum(K)=W_sum(K)+W(X,XX);
                W_count(K)=W_count(K)+1;
            end
        end
    end
end

%% computate average
for K=1:1:k
    W_average(K)=W_sum(K)/W_count(K);
end

%% sort k
[W_average numR]=sort(W_average);
for K=1:1:k
    for X=1:1:size(W_k,1)
        for XX=1:1:size(W_k,2)
            if W_k(X,XX)==numR(K)
                W_k(X,XX)=K;
            end
        end
    end
end

%% computate distance from k
W_distance=zeros(size(W,1),size(W,2),k);
for K=1:1:k
    for X=1:1:size(W_k,1)
        for XX=1:1:size(W_k,2)
            W_distance(X,XX,K)=abs((W_average(K)-W(X,XX)));
        end
    end
end

%% Regroup

for X=1:1:size(W_k,1)
    for XX=1:1:size(W_k,2)
        [Num,NumK]=min(W_distance(X,XX,:));
%         if NumK>W_k(X,XX)
%             W_k(X,XX)=W_k(X,XX)+1;
%         elseif NumK<W_k(X,XX)
%             W_k(X,XX)=W_k(X,XX)-1;
%         end
        
        W_k(X,XX)=NumK;
    end
end

%% sort k
[W_average numR]=sort(W_average);
for K=1:1:k
    for X=1:1:size(W_k,1)
        for XX=1:1:size(W_k,2)
            if W_k(X,XX)==numR(K)
                W_k(X,XX)=K;
            end
        end
    end
end

end

end

