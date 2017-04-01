function [grads] = gradient_5layer(W1,W2,W3,W4,W5,b1,b2,b3,b4,b5,x,t,ReLU)
%GRADIENT 
%   

batch_size=size(x,1);

%forward
%% layer 1
%neuron ops.
a1=x*W1;
for X=1:1:size(a1,1)
    a1(X,:)=a1(X,:)+b1;
end
%activation function
if ReLU==1
else
[z1]=sigmoid(a1);
end

%% layer 2
a2=z1*W2;
for X=1:1:size(a2,1)
    a2(X,:)=a2(X,:)+b2;
end
[z2]=sigmoid(a2);

%% layer 3
a3=z2*W3;
for X=1:1:size(a3,1)
    a3(X,:)=a3(X,:)+b3;
end
[z3]=sigmoid(a3);

%% layer 4
a4=z3*W4;
for X=1:1:size(a4,1)
    a4(X,:)=a4(X,:)+b4;
end
[z4]=sigmoid(a4);

%% layer 5
a5=z4*W5;
for X=1:1:size(a5,1)
    a5(X,:)=a5(X,:)+b5;
end

y=zeros(size(a5,2));
for X=1:1:size(a5,1)
y(X,:)=softmax(a5(X,:));
end

grads.loss=cross_entropy(y,t);


%% backward
%% layer 5
dy=(y-t)/batch_size;
grads.W5=(z4.'*dy);
for X=1:size(dy,2)
    grads.b5(X)=sum(dy(:,X));
end


%% layer 4
da4=(dy*W5.');
dz4=zeros(size(a4));

for X=1:1:size(a4,1)
    for XX=1:1:size(a4,2)
        dz4(X,XX)=(1-sigmoid(a4(X,XX)))*sigmoid(a4(X,XX))*da4(X,XX);
    end
end

grads.W4=(z3.'*dz4);
for X=1:size(dz4,2)
    grads.b4(X)=sum(dz4(:,X));
end

%% layer 3
da3=(dz4*W4.');
dz3=zeros(size(a3));

for X=1:1:size(a3,1)
    for XX=1:1:size(a3,2)
        dz3(X,XX)=(1-sigmoid(a3(X,XX)))*sigmoid(a3(X,XX))*da3(X,XX);
    end
end

grads.W3=(z2.'*dz3);
for X=1:size(dz3,2)
    grads.b3(X)=sum(dz3(:,X));
end

%% layer 2
da2=(dz3*W3.');
dz2=zeros(size(a2));

for X=1:1:size(a2,1)
    for XX=1:1:size(a2,2)
        dz2(X,XX)=(1-sigmoid(a2(X,XX)))*sigmoid(a2(X,XX))*da2(X,XX);
    end
end

grads.W2=(z1.'*dz2);
for X=1:size(dz2,2)
    grads.b2(X)=sum(dz2(:,X));
end

%% layer 1
da1=(dz2*W2.');
dz1=zeros(size(a1));

for X=1:1:size(a1,1)
    for XX=1:1:size(a1,2)
        dz1(X,XX)=(1-sigmoid(a1(X,XX)))*sigmoid(a1(X,XX))*da1(X,XX);
    end
end

grads.W1=(x.'*dz1);
for X=1:size(dz1,2)
    grads.b1(X)=sum(dz1(:,X));
end

grads.dz1=dz1;

end

