function [loss2] = loss_func(x,W,t)
%LOSS ���̊֐��̊T�v�������ɋL�q
%   �ڍא����������ɋL�q
z1=x*W;
y=softmax(z1);
loss2=cross_entropy(y,t);

end

