function[X,y] = SimGenerate(num,p,SS)

X = randn(num,p);
X= X * SS;


y = 4*X(:,98).*X(:,99).*X(:,97) + 2.3*exp(-X(:,99)) + 4*X(:,100) + 0.9*X(:,96).^3 +randn(num,1);

y = y -mean(y);


