function [ acc_test, acc_test_locals ] = locals_balanced_ELM_v2_test( testdata,testlabel,nnall,a,minIndex )
%
%   locals_balanced_ELM_test responds on completing the classification mission.
%
%   Inputs:
%       testdata : original dataset whose size is [M,N], which means feature Dimens is M and the original dataset contains N samples;
%       testlabel: the label corresponds to the testdata;
%       a        : the weights of locals_ELMs;
%       minindex : the  corresponds locals_ELM model series to the weights above;
%       nnall    : all locals_ELM's neural networks parameters;
%
%    Outputs:
%       acc_test : the accuracy of test dataset for the locals_balanced_ELM.

ndata = length(testlabel);
T = 0;
nn = nnall{1};
[testdata_cell,~] = locals_feature_initializer(testdata, testlabel, nn.Cons_in);
for i = 1:length(nnall)
    %% -----------substract the corresponding neural network parameters----------
    nn = nnall(minIndex(i));
    [tempnn,~] = elm_test(testdata_cell{i},testlabel,nn{1});
    acc_test_locals(i) = mean(tempnn.label_actual == testlabel);
    tempT = zeros(max(testlabel),ndata);
    %% ------------calculate the actual output of the assambled locals_balanced_ELM iterably---------
    for j = 1:ndata
        tempT(tempnn.label_actual(j),j) = 1;
    end
    T = a(i)*tempT+T;
end
%% ------------calculate the accuracy of test dataset--------------
[~,actual_label] = max(T);
acc_test = sum(actual_label==testlabel)/ndata;
end

