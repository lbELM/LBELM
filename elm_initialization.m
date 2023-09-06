function nn = elm_initialization(nn)
%
%   elm_initilization responds on completing the intilization mission of the stardand ELM model.
%
%   Inputs:
%       nn  : the neural network parameters.
%
%    Outputs:
%       nn  : the neural network parameters.

%% ----------------generate biases and input weights randomly----------------------
scale = 1;

nn.b = 2*scale*rand(nn.hiddensize,1,'single')-scale;
nn.W = 2*scale*rand(nn.hiddensize, nn.inputsize,'single')-scale;

% nn.b = randn(nn.hiddensize,1);
% nn.W = randn(nn.hiddensize, nn.inputsize);

% if nn.orthogonal
%     if nn.hiddensize > nn.inputsize
%         nn.W = orth(nn.W);
%     else
%         nn.W = orth(nn.W')';
%     end
%     nn.b=orth(nn.b);
% end

end

