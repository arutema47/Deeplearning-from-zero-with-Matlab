W=randn(2,3);
u=1e-1

%% 学習ループ
for iter=1:1:1000

x=[0.6 0.9];

t=[0 0 1]; %正解ラベル

loss_out(iter)=loss_func(x,W,t);

[grad] = numerical_gradient(x,W,t); %勾配導出

W=W-grad*u; %学習
end

semilogy(loss_out)