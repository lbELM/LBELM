function [a,minIndex,nnall, Time] = locals_balanced_ELM_v2_train(traindata, trainlabel,  nn)
%
%   locals_balanced_ELM_train responds on iterating the precedure of Adaboost.
%
%   Inputs:
%       traindata : original dataset whose size is [M,N], which means feature Dimens is M and the original dataset contains N samples;
%       trainlabel: the label corresponds to the traindata;
%       testdata  : original dataset whose size is [M,N], which means feature Dimens is M and the original dataset contains N samples;
%       testlabel : the label corresponds to the testdata;
%       Cons_in   : the construction of dataSet; i. e. [M/3 M/3 M/3]. 
%
%    Outputs:
%       a         : the weights of locals_ELMs;
%       minindex  : the  corresponds locals_ELM model series to the weights above;
%       nnall     : all locals_ELM's neural networks parameters;
%       T         : the time consumption of this procedure.

T_start = clock;
%% ------------------------------------the precedure of initialization----------------------------------------------
%
% --------------initilize the locals_ELM----------------------
nnall=locals_ELM_initializer(traindata, trainlabel, nn.Cons_in, nn);

% -----------summarize the classification results-------------------
N = length(trainlabel);
H=zeros(N, length(nnall));
for i=1:3
    nnall{i}.label_actual=nn.label_actual; %123
end
for i=1:length(nnall)
    H(:,i)=nnall{i}.label_actual;
end
% -------summarize the index matrix of correct classification and the wrong classification-----------------
errDataAll={};
accDataAll={};
for i=1:length(nnall)
    suberrdataH=find(H(:,i)'~=trainlabel);
    subaccdataH=find(H(:,i)'==trainlabel);
%     if sum(size(suberrdataH)) == 1
%         disp(['locals_ELM',num2str(i),'reaches the acuracy 100%'])
%         break;
%     end
    errDataAll=[errDataAll,suberrdataH];    
    accDataAll=[accDataAll,subaccdataH];
end

% -----------------initialize the weights of samples------------------
D{1} = zeros(N,1)+1/N;
% --------------initilize the weight of the locals_ELMs----------------
a = zeros(length(nnall),1);
%% --------------------------------------the precedure of Adaboost---------------------------------
minIndex = [];%minIndex represents the series of the locals_ELM which holds the minimum error
for j=1:length(nnall)
% ------------calculate the error of the locals_ELMs---------------
    errAll=[];
    D_hat = D{j};
    for i=1:length(nnall)
        err=sum(D_hat(errDataAll{i}));
        errAll=[errAll,err];
    end
    [minErr,minIndex(j)]=min(errAll);
    % -------------------------the iteration precedure of the locals_ELM's weights-------------------------------------
    a(j)=0.5*log((1-minErr)/minErr);%calculate the weights of the chosen locals_ELM
    % ----------------------------the iteration procedure of the samples weights--------------------------------------
    minErrData=errDataAll(:,minIndex(j));%summarize the sample index of wrong classification for the chosen locals_ELM
    minAccData=accDataAll(:,minIndex(j));%summarize the sample index of correct classification for the chosen locals_ELM
    % -------------decrease the weights of the correct-classification ssamples--------------------------
    for i=minAccData{1}'
        D_hat(i)=D_hat(i)/(2*(1-minErr));
    end
% -----------  ---increase the weights of the wrong-classification ssamples---------------------------
    for i=minErrData{1}'
         D_hat(i)=D_hat(i)/(2*minErr);
    end
    D{j+1} = D_hat;
end
T_end = clock;
Time = etime(T_end, T_start);

end













    
    