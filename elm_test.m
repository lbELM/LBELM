function [nn, acc_test] = elm_test(X,Y,nn)

ndata        = size(X, 2);
tempH        = nn.W*X + repmat(nn.b,1,ndata);

switch lower(nn.activefunction)
    case{'s','sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
    case{'t','tanh'}
        H = tanh(tempH);
end

Y_hat    = nn.beta*H;

clear H;
acc_test = [];
if ismember(nn.type,{'c','classification','Classification'})
    [~,label_actual]  = max(Y_hat,[],1);
    
    nn.label_actual=label_actual;
    
    if ~isempty(Y)
        [~,label_desired] = max(Y,[],1);
        acc_test = sum(label_actual==label_desired)/ndata;
        acc_3action=[];
        for i=sort(unique(label_desired))
            subacc=sum(label_actual(label_desired==i)==i)/sum(label_desired==i);
            acc_3action=[acc_3action,subacc];
        end
        nn.acc_3action=acc_3action;
    end
else
    if ~isempty(Y)
        normfro   = norm(Y-Y_hat,'fro');
        acc_test = sqrt(normfro^2/ndata);
    end
end

nn.testlabel  = Y_hat;
if ~isempty(Y)
    nn.acc_test   = acc_test;
end



