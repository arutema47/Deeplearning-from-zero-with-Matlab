function [grads] = gradient(W1,W2,W3,b1,b2,b3,x,t)
%GRADIENT この関数の概要をここに記述
%   詳細説明をここに記述

batch_size=size(x,1);

%forward
a1=x*W1;
for X=1:1:size(a1,1)
    a1(X,:)=a1(X,:)+b1;
end
% [z1,mask1]=ReLU(a1);
[z1]=sigmoid(a1);

a2=z1*W2;
for X=1:1:size(a2,1)
    a2(X,:)=a2(X,:)+b2;
end
[z2]=sigmoid(a2);

a3=z2*W3;
for X=1:1:size(a3,1)
    a3(X,:)=a3(X,:)+b3;
end

y=zeros(size(a3,2));
for X=1:1:size(a3,1)
y(X,:)=softmax(a3(X,:));
end

grads.loss=cross_entropy(y,t);



%backward
dy=(y-t)/batch_size;
grads.W3=(z2.'*dy);
for X=1:size(dy,2)
    grads.b3(X)=sum(dy(:,X));
end

da2=(dy*W3.');

%layer2 back
dz2=zeros(size(a2));

for X=1:1:size(a2,1)
    for XX=1:1:size(a2,2)
        dz2(X,XX)=(1-sigmoid(a2(X,XX)))*sigmoid(a2(X,XX))*da2(X,XX);
    end
end

% dz1size=size(dz1);

grads.W2=(z1.'*dz2);
for X=1:size(dz2,2)
    grads.b2(X)=sum(dz2(:,X));
end

da1=(dz2*W2.');
dz1=zeros(size(a1));


for X=1:1:size(a1,1)
    for XX=1:1:size(a1,2)
        dz1(X,XX)=(1-sigmoid(a1(X,XX)))*sigmoid(a1(X,XX))*da1(X,XX);
    end
end

% dz1size=size(dz1);

grads.W1=(x.'*dz1);
for X=1:size(dz1,2)
    grads.b1(X)=sum(dz1(:,X));
end

end

