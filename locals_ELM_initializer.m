function [nnall] = locals_ELM_initializer(traindata, trainlabel, Cons_in, nn)
%
%   locals_ELM_initilizer method generates some locals_ELM model.
%
%
%   Inputs:
%       traindata  : original dataset whose size is [M,N],which means feature Dimens is M and the original dataset contains N samples;
%       trainlabel : the label corresponds to the traindata;
%       testdata   : original dataset whose size is [M,N], which means feature Dimens is M and the original dataset contains N samples;
%       testlabel  : the label corresponds to the testdata;
%       Cons_in    : the construction of dataSet; i. e. [M/3 M/3 M/3].
%
%    Outputs:
%       nnall      : all locals_ELM's neural networks parameters.

rng('default'); %set the random seed


%% -------locals feature initiate--------

[traindata_cell,trainlabel_cell] = locals_feature_initializer(traindata, trainlabel, Cons_in);
nnall = {};
%% -----convert the label to the onehot type---------------
for i = 1:length(Cons_in)
    trainlabel_1cell{i} = label_convert(trainlabel_cell, '1');
%     testlabel_1cell{i}= label_convert(testlabel_cell{i}, '1');
    nn.inputsize        =size(traindata_cell{i},1);%feature's size
%% -------initialization--------
    nn = elm_initialization(nn);

%% --------RELM---------


    [nn, acc_train]  = elm_train(traindata_cell{i}, trainlabel_1cell{i}, nn);
    nn.locals        = ['local_', num2str(i)];
    [~, acc_test]    = elm_test(traindata_cell{i}, trainlabel_1cell{i}, nn);  
    nn.acc_train     = acc_train;
    nn.acc_test      = acc_test;
    nnall            = [nnall, {nn}];
end
end






