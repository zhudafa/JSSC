function X = proj_l20(B,card)

% min 0.5*||X-B||_F^2
% s.t. ||X||_{2,0} <= card

y = [];
X = zeros(size(B));
for i = 1:size(B,1)
    yi = norm(B(i,:)); 
    y = [y;yi];
end

[~, position] = maxk(y,card); 
X(position,:) = B(position,:); 



