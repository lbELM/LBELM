
function [data_out,label_out] = locals_feature_initializer(data_in, label_in, Cons_in)
%
%   locals_feature_initilizer method generates some subsets of the input data.
%
%   Inputs:
%       data_in : original dataset whose size is [M,N],which means feature Dimens is M and the original dataset contains N samples;
%       label_i: the label corresponds to the input data;
%       Cons_in : the construction of dataSet; i. e. [M/3 M/3 M/3]. 
%
%    Outputs:
%       data_out : A locals-feature set  whose size is [M/3, N], where A = size(C_in, 2);
%       label_out: the label corresponds to the output data.

[M, ~] = size(data_in);
%% -----------------------check the input C_in------------------------
if (exist('Cons_in')~=1)
    error('Cons_in no assign')  
end

if sum(Cons_in) ~= M
    error('C_in is not compatible to the M')
end
%% --------------------split data_in for the local features-----------------------
data_out = {};

%for i = size(Cons_in, 2)
%   if i == 1
%       data_out{i} = data_in(1:Cons_in(i), :);
%   end
%   data_out{i} = data_in(sum(Cons_in(1:i-1))+1:sum(Cons_in(1:i)), :);
%end
for i=1:3
   
   data_out{i} = data_in(sum(Cons_in(1:i-1))+1:sum(Cons_in(1:i)), :);
end
label_out = label_in;

end
