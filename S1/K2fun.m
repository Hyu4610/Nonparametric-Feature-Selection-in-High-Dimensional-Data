% TK is the pre calculated pairwise distance ready for calculate kernel matrix.

function K2 = K2fun(TK,lambda, p)

K2=1+lambda(1)*TK{1};
if (p>1)
for j = 2:1:p
    K2=K2.*(1+lambda(j)*TK{j});
end
end

    
