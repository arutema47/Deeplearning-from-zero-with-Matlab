W=randn(2,3);
u=1e-1

%% �w�K���[�v
for iter=1:1:1000

x=[0.6 0.9];

t=[0 0 1]; %�������x��

loss_out(iter)=loss_func(x,W,t);

[grad] = numerical_gradient(x,W,t); %���z���o

W=W-grad*u; %�w�K
end

semilogy(loss_out)