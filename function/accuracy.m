function [accuracy] = accuracy(x,t,W1,b1,W2,b2)
%ACCURACY この関数の概要をここに記述
%   詳細説明をここに記述
%forward
a1=x*W1;
for X=1:1:size(a1,1)
    a1(X,:)=a1(X,:)+b1;
end
[z1]=sigmoid(a1);

a2=z1*W2;
for X=1:1:size(a2,1)
    a2(X,:)=a2(X,:)+b2;
end

y=zeros(size(a2,2));
for X=1:1:size(a2,1)
y(X,:)=softmax(a2(X,:));
end

[xxx,y]=max(y.');
y=y.';

[xxx,t]=max(t.');
t=t.';

accuracy=sum(y==t)/size(x,1);


end

