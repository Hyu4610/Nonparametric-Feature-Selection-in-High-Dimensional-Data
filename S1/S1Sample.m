%numTrain, numTest are the sample sizes for training data and testing data. p is the dimension. TTR,TTL are the separate testing data covariate and testing label. TR, TL are data for the training and
%tuning parameters. gamma1 and gamma2 are the tuning parameters. rrblah is the random seed.

%generate data
rng(1000);
numTrain = 100;
numTest = 10000;
p = 100;
rrblah = 900;

%correlation matrix
RR1=[1,0.4,-0.3,0;
    0.4,1,0.5,0;
    -0.3,0.5,1,0.2;
    0,0,0.2,1];
    
RR = [eye(p-4),zeros(p-4,4);zeros(4,p-4),RR1];
SS=sqrtm(RR);   

[TTR,TTL] = SimGenerate(numTest,p,SS);


E0 = [];
F0=[];

for rr = rrblah
    rng(rr);

    [TR,TL] = SimGenerate(numTrain,p,SS);

    cvIndices = crossvalind('Kfold',numTrain,3);
    
    for ic = 1:16
        gamma1 = 2^(-17+2*ic);
        
        for jc = 1:16
             gamma2 = 2^(-17+2*jc);

             
              CLM = [];
              
               for ik = 1:3
                   testind = (cvIndices == ik); 
                   trainind = ~testind;
                   train_data = TR(trainind,:);
                   X = train_data;
                   Y = TL(trainind,:);
                   
                   test_data = TR(testind,:);
                   test_label = TL(testind,:);

                   % CL is prediction error.
              
                   [~,CL] = maincodefun(X,Y,test_data,test_label, gamma1, gamma2,p);
                
                   
                   CLM = [CLM;CL];
               end
               F0 = [F0;gamma1,gamma2,mean(CLM),rr];
               
             
        end
        
    end
    
    E111 = sortrows(F0,3);
    gamma1 = E111(1,1);
    gamma2 = E111(1,2);
    X = TR;
    Y = TL;
    test_data =TTR;
    test_label = TTL;
    

    [lambda,CL,k,Converflag] = maincodefun(X,Y,test_data,test_label, gamma1, gamma2,p);
    E0 = [E0;gamma1,gamma2,lambda,CL,k, Converflag,rr];
     
end

E0_rr_rrblah = E0;
save('resultsdirblah/Proposed0608/n=100 p=100/E0_rr_rrblah.mat','E0_rr_rrblah')

