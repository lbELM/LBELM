%=========================================================================
%%%    Authors      : Yang Liu
%%%    Institute    : Hangzhou Dianzi University, CHINA
%%%    Email        : ly19950119@hdu.edu.cn
%%%    Date         : NOV 2019
%--------------------------------------------------------------------------

clc
clear
%% ---------------------------------------------Load data stage----------------------------------------------


loadfile = 'E:\EA-SRC\LBELM_data\EEG_EMG_LBELM_train_1.mat';
load(loadfile);
traindata = single(data(1:60, :)');
trainlabel = labels(1:60)' + 1;

loadfile = 'E:\EA-SRC\LBELM_data\EEG_EMG_LBELM_test_1.mat';
load(loadfile);
testdata = single(data(1:60, :)');
testlabel = labels(1:60)' + 1;
clear data
clear labels



%% -------setting----------
nn.Cons_in         = [14 14 14];
nn.C               = exp(2:2:10);%the regularization coefficient
nn.hiddensize      =3000;%the number of hidden unit

nn.activefunction ='s';%the 's' represents sigmoid and the 't' represents tanh
nn.method         ='RELM';%'ELM' means standard ELM, 'RELM' means regularized ELM
nn.type           ='classification';%'classification' symbolizes the objective of classifying, others words symbolizes the objective of regression
nn.label_actual   =testlabel;   %...
%% -------------------------------locals_balanced_ELM training stage-------------------------

[a,minIndex,nnall, Time] = locals_balanced_ELM_v2_train(traindata, trainlabel, nn);



fprintf('      method      |      locals       |    Optimal C    |  Training Acc.  |    Testing Acc.   |   Training Time \n');
fprintf('--------------------------------------------------------------------------------------------\n');


%% -------------------------------locals_balanced_ELM testing stage-------------------------
[acc_train, ~] = locals_balanced_ELM_v2_test(traindata,trainlabel,nnall,a,minIndex);
[acc_test, acc_test_locals] = locals_balanced_ELM_v2_test(testdata,testlabel,nnall,a,minIndex);
for i=1:length(nnall)  
    nn_temp = nnall{i};
    fprintf('      %6s      |      %6s      |     %.5f     |      %.3f      |      %.5f      |      %.5f      \n', nn_temp.method, nn_temp.locals, nn_temp.C_opt, nn_temp.acc_train, acc_test_locals(i), nn_temp.time_train);    
end
fprintf('      %6s      |      %6s       |     %.5s              |      %.3f      |      %.5f      |      %.5f      \n', 'lbELM', 'fusion', 'None', acc_train, acc_test, Time);    
