function [beta, optLambda, LOO] = regressor(H, Y, lamdas)
%
%   regression responds on completing the procedure of LOO optimization based on MSE.
%
%   Inputs:
%       H      : the hidden layer output matrix;
%       Y      : the laebl of traindata. 
%
%    Outputs:
%       beta              : the output weight matrix of hidden layer;
%       optLambda         : the optimal lambda;
%       LOO               : the summary of MSE based on PRESS.

if nargin == 2
    lamdas = exp(-7:1:7);
end

nData = size(H,1);

if numel(lamdas) == 1
    optLambda = lamdas; LOO = Inf;
    if nData < size(H,2)
        beta = H'*pinv(H*H'+optLambda*eye(nData))*Y;
    else
        beta = pinv(H'*H+optLambda*eye(size(H,2)))*H'*Y;
    end
else
    LOO  = inf(1,numel(lamdas));
   %% -------------------------calculate the MSE based on the PRESS -------------------
    if nData < size(H,2)
        %% ---------------precalculate (N < L)----------------------
        HH     = H*H';
        [U, S] = svd(HH);
        S      = diag(S)';
        A      = HH*U;
        B      = U'*Y;
        %% --------------calculate the MSE based on PRESS by iterating the candidate lambda---------------
        for iLamda = 1 : length(lamdas)
            lamdaCur   = lamdas(iLamda);
            temp    = A.*repmat(1./(S+lamdaCur),length(S),1);
            HAT     = sum(temp.*U,2);
            Y_hat   = temp*B;
            errDiff = (Y-Y_hat)./repmat((1-HAT),1,size(Y,2));
            normFro = norm(errDiff,'fro');
            errLoo  = normFro^2/nData;
            LOO(iLamda)  = errLoo;
        end
        %% ---------------calculate the optimal beta---------------------------
        [~,ind]  = min(LOO);
        optLambda = lamdas(ind(1));
        beta     = H'*(U.*repmat(1./(S+optLambda),length(S),1))*B;
    else
        %% ---------------precalculate (N >= L)----------------------
        [U, S] = svd(H'*H);
        S      = diag(S)';
        A      = H*U;
        B      = A'*Y;
        %% --------------calculate the MSE based on PRESS by iterating the candidate lambda---------------
        for iLamda = 1 : length(lamdas)
            lamdaCur= lamdas(iLamda);
            temp    = A.*repmat(1./(S+lamdaCur),size(A,1),1);
            HAT     = sum(temp.*A,2);
            Y_hat   = temp*B;
            errDiff = (Y-Y_hat)./repmat((1-HAT),1,size(Y,2));
            normFro = norm(errDiff,'fro');
            errLoo  = normFro^2/nData;
            LOO(iLamda)  = errLoo;
        end
        %% ---------------calculate the optimal beta---------------------------
        [~,ind]  = min(LOO);
        optLambda = lamdas(ind(1));
        beta     = U.*repmat(1./(S+optLambda),length(S),1)*B;
    end
end





