function [x_proj, supp] = projectVec(x,A,y,k)
%function [x_proj, supp] = projectVec(x,A,y,k)
[~,I] = sort(abs(x),'descend');
supp = I(1:k);

x_proj = zeros(numel(x),1);
x_proj(supp) = A(:,supp)\y;
end
