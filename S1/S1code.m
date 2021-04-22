addpath('maindirblah')
%generate data
rng(1000);
numTrain = 100;
numTest = 10000;
p = 100;
xxx1 = randn(numTest,p);

RR1=[1,0.4,-0.3,0;
    0.4,1,0.5,0;
    -0.3,0.5,1,0.2;
    0,0,0.2,1];
    
RR = [eye(p-4),zeros(p-4,4);zeros(4,p-4),RR1];
SS=sqrtm(RR);   
xxx1=xxx1 * SS;
yy = 4*xxx1(:,98).*xxx1(:,99).*xxx1(:,97) + 2.3*exp(-xxx1(:,99)) + 4*xxx1(:,100) + 0.9*xxx1(:,96).^3 +randn(numTest,1);

yy = yy- mean(yy);
TTR = xxx1;
TTL = yy;


E0 = [];
F0=[];

for rr = rrblah
    rng(rr);
    
    x1 = randn(numTrain,p);
    
    x1=x1 * SS;
   
  
    y = 4*x1(:,98).*x1(:,99).*x1(:,97) + 2.3*exp(-x1(:,99))+ 4*x1(:,100) + 0.9*x1(:,96).^3 +randn(numTrain,1);

    y = y - mean(y);
    

    TR = x1;
    TL = y;
    
    cvIndices = crossvalind('Kfold',numTrain,3);
    
    for ic = 1:16
        gamma1 = 2^(-17+2*ic);
        
        for jc = 1:16
             gamma3 = 2^(-17+2*jc);
             gamma2 = 0.001*gamma3;
             
              CLM = [];
              
               for ik = 1:3
                   testind = (cvIndices == ik); 
                   trainind = ~testind;
                   train_data = TR(trainind,:);
                   X = train_data;
                   Y = TL(trainind,:);
                   
                   test_data = TR(testind,:);
                   test_label = TL(testind,:);
              
                   [~,CL] = maincodefun0608(X,Y,test_data,test_label, gamma1, gamma2, gamma3,p);
                
                   
                   CLM = [CLM;CL];
               end
               F0 = [F0;gamma1,gamma3,mean(CLM),rr];
               
             
        end
        
    end
    
    E111 = sortrows(F0,3);
    gamma1 = E111(1,1);
    gamma3 = E111(1,2);
    gamma2 = 0.001*gamma3; 
    X = TR;
    Y = TL;
    test_data =TTR;
    test_label = TTL;
    

    [lambda,CL,k,Converflag] = maincodefun0608(X,Y,test_data,test_label, gamma1, gamma2, gamma3,p);
    E0 = [E0;gamma1,gamma3,lambda,CL,k, Converflag,rr];
     
end

E0_rr_rrblah = E0;
save('resultsdirblah/Proposed0608/n=100 p=100/E0_rr_rrblah.mat','E0_rr_rrblah')

