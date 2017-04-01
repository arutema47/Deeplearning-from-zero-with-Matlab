% %% Obtain image
% train = csvread('train.csv', 1, 0);                  % read train.csv
% test = csvread('test.csv', 1, 0);                  % read test.csv
% 
% 
% 
% 
clear all
load('mnist_3.mat')
% 'completed loading'
% 
% %% Prepare data
% n = size(images_train, 1);                    % number of samples in the dataset
% targets  = labels_train;                 % 1st column is |label|
% targets(targets == 0) = 10;         % use '10' to present '0'
% targetsd = dummyvar(targets);       % convert label into a dummy variable
% inputs = images_train;               % the rest of columns are predictors
% 
% % inputs = inputs';                   % transpose input
% targets = targets';                 % transpose target
% targetsd = targetsd';               % transpose dummy variable
% 
% %% test
% n = size(images_test, 1);                    % number of samples in the dataset
% targets_test  = labels_test;                 % 1st column is |label|
% targets_test(targets_test == 0) = 10;         % use '10' to present '0'
% targetsd_test = dummyvar(targets_test);       % convert label into a dummy variable
% inputs_test = images_test;               % the rest of columns are predictors
% 
% % inputs_test = inputs_test';                   % transpose input
% targets_test = targets_test';                 % transpose target
% targetsd_test = targetsd_test';               % transpose dummy variable
% 
% rng(1);                             % for reproducibility

'prepared data'

% x=rand(100,784);
% t=rand(100,10);



%% Parameters
ITER=10000;
input_size=784;
batch_size=100;
hidden_size=50;
output_size=10;
weight_init_std=0.2;
u=0.1;
iter_per_epoch=max(size(inputs,2)/batch_size);
ECOUNT=iter_per_epoch;
EPOCH=0;

%% INIT
W1=sqrt(2/hidden_size)*randn(input_size,hidden_size);
b1=zeros(1,hidden_size);
W2=sqrt(2/hidden_size)*randn(hidden_size,output_size);
b2=zeros(1,output_size);

for COUNT=1:1:ITER

    %% mini batch
       % determine how many elements is ten percent
   % get the randomly-selected indices
   indices = randperm(length(inputs));
   indices = indices(1:batch_size);
   % choose the subset of a you want
   x_batch = inputs(:,indices).';
   t_batch = targetsd(:,indices).';


    %% Grads
% conv. grads
%     [grad1] = numerical_gradient_1(W1,W2,b1,b2,x,t);
%     [grad2] = numerical_gradient_2(W1,W2,b1,b2,x,t);

%back propagation
    [grad1] = gradient(W1,W2,b1,b2,x_batch,t_batch); %high-speed
    
    %% Feedback
    W1=W1-grad1.W1*u;
    W2=W2-grad1.W2*u;
    b1=b1-grad1.b1*u;
    b2=b2-grad1.b2*u;
    


loss_DNN(COUNT)=grad1.loss;
figure(1)
plot(loss_DNN)
drawnow


if ECOUNT==iter_per_epoch
ECOUNT=1;

   indices = randperm(length(inputs_test));
   indices = indices(1:batch_size*10);
   % choose the subset of a you want
   x_test = inputs_test(:,indices).';
   t_test = targetsd_test(:,indices).';

   indices = randperm(length(inputs_test));
   indices = indices(1:batch_size*10);
   % choose the subset of a you want
   x_batch = inputs(:,indices).';
   t_batch = targetsd(:,indices).';

[accuracy_train(EPOCH+1)] = accuracy(x_batch,t_batch,W1,b1,W2,b2);
[accuracy_test(EPOCH+1)] = accuracy(x_test,t_test,W1,b1,W2,b2);
figure(2)
plot(0:1:EPOCH,accuracy_train,0:1:EPOCH,accuracy_test)
drawnow

%% show EPOCH
disp('EPOCH')
disp(EPOCH)
disp('train accracy ' )
disp(accuracy_train(EPOCH+1)*100)
disp('test accracy ' )
disp(accuracy_test(EPOCH+1)*100)
EPOCH=EPOCH+1;

end
ECOUNT=ECOUNT+1;

end