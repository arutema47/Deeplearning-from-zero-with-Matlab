function [loss2] = loss_func(x,W,t)
%LOSS この関数の概要をここに記述
%   詳細説明をここに記述
z1=x*W;
y=softmax(z1);
loss2=cross_entropy(y,t);

end

