function [LabelNew] = label_convert(Label, type)
%
%   label_convert responds on converting the label to the onehot type.
%
%   Inputs:
%       Label  : original dataset whose size is [M,N], which means feature Dimens is M and the original dataset contains N samples;
%       type   : the onehot type. '1' means the [-1 -1 1] type; '2' means the [0 0 1] type.
%
%    Outputs:
%       LabelNew : the onehot label which holds a coincidence to the type.

if nargin < 2
    type = '2';
end
%% -----------figure out the informance of Label-----------
classes    = unique(Label);
nClasses   = numel(classes);
nData      = numel(Label);


LabelNew = -ones(nClasses,nData,'single');

%% ----------complete the procedure of conversion--------------
for i = 1 : nClasses
    LabelNew(i,Label==classes(i)) = 1;
end

if ~strcmp(type,'2')
    LabelNew = (LabelNew+1)/2;
end


end

