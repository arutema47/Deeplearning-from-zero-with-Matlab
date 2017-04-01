function [accuracy] = accuracy(W1,W2,W3,W4,W5,b1,b2,b3,b4,b5,x,t)
%ACCURACY この関数の概要をここに記述
%   詳細説明をここに記述
batch_size=size(x,1);

%forward
%% layer 1
a1=x*W1;
for X=1:1:size(a1,1)
    a1(X,:)=a1(X,:)+b1;
end
%[z1,mask1]=ReLU(a1);
[z1]=sigmoid(a1);
%% layer 2
a2=z1*W2;
for X=1:1:size(a2,1)
    a2(X,:)=a2(X,:)+b2;
end
%[z2,mask2]=ReLU(a2);
[z2]=sigmoid(a2);

%% layer 3
a3=z2*W3;
for X=1:1:size(a3,1)
    a3(X,:)=a3(X,:)+b3;
end
%[z3,mask3]=ReLU(a3);
[z3]=sigmoid(a3);

%% layer 4
a4=z3*W4;
for X=1:1:size(a4,1)
    a4(X,:)=a4(X,:)+b4;
end
%[z4,mask4]=ReLU(a4);
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

[xxx,y]=max(y.');
y=y.';

[xxx,t]=max(t.');
t=t.';

accuracy=sum(y==t)/size(x,1);


end

