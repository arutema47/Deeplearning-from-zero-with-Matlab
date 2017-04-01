function [gradW gradB loss] = gradient(W,b,x,t)
%GRADIENT この関数の概要をここに記述
%   詳細説明をここに記述

batch_size=size(x,1);
layer_size=size(W,3);

% forward propagation
for LAYER=1:1:layer_size
    %Neuron function
    if LAYER==1
        a(:,:,LAYER)=x*W(:,:,LAYER);
    else
        a(:,:,LAYER)=z(:,:,LAYER-1)*W(:,:,LAYER);
    end
    
    for X=1:1:size(a(:,:,LAYER),1)
        a(X,:,LAYER)=a(X,:,LAYER)+b(1,:,LAYER);
    end
    
    %activation function
    if LAYER==layer_size
        y=zeros(size(a(:,:,LAYER),2));
        for X=1:1:size(a(:,:,LAYER),1)
            y(X,:)=softmax(a(X,:,LAYER));
        end
    else
        [z(:,:,LAYER)]=sigmoid(a(:,:,LAYER));
    end
    
    % cross entropy
    loss=cross_entropy(y,t);
end


% backward propagation
dy=(y-t)/batch_size;
for LAYER=layer_size:-1:1
    if LAYER==layer_size
        gradW(:,:,LAYER)=(z(:,:,LAYER-1).'*dy);
        for X=1:size(dy,2)
            gradB(1,X,LAYER)=sum(dy(:,X));
        end
        da(:,:,LAYER-1)=(dy*W(:,:,LAYER).');
        for X=1:1:size(a(:,:,LAYER),1)
            for XX=1:1:size(a(:,:,LAYER),2)
                dz(X,XX,LAYER-1)=(1-sigmoid(a(X,XX,LAYER-1)))*sigmoid(a(X,XX,LAYER-1))*da(X,XX,LAYER-1);
            end
        end
        
        
    elseif LAYER==1
        gradW(:,:,LAYER)=x.'*dz(:,:,LAYER);
        for X=1:size(dy,2)
            gradB(1,X,LAYER)=sum(dz(:,X,LAYER));
        end
        
    else
        gradW(:,:,LAYER)=(z(:,:,LAYER-1).'*dz(:,:,LAYER));
        for X=1:size(dz,2)
            gradB(1,X,LAYER)=sum(dz(:,X,LAYER));
        end
        da(:,:,LAYER-1)=(dz(:,:,LAYER)*W(:,:,LAYER).');
        for X=1:1:size(a(:,:,LAYER),1)
            for XX=1:1:size(a(:,:,LAYER),2)
                dz(X,XX,LAYER-1)=(1-sigmoid(a(X,XX,LAYER-1)))*sigmoid(a(X,XX,LAYER-1))*da(X,XX,LAYER-1);
            end
        end
    end
end






end

