% This function is the main function for the algorithm. (X,Y) is for the training. (test_data, test_labe) is for % testing. gamma1, gamma2 are the tuning parameter. p is the dimension.

function [lambda,CL,k, Converflag] = maincodefun(X,Y, test_data, test_label, gamma1, gamma2, p)

                   numTest = size(test_data,1);
                   numTrain = size(X,1);
                   
 
                   %sigma
                   temp=X*X';
                   temp2=diag(temp);
                   Q=repmat(temp2,1,numTrain)+repmat(temp2',numTrain,1)-2*temp;
                   
                   sigma = 1/(2*median(sort(Q(:)))); % sigma consistent with package like svm
                
                   
                   CO= [abs(corr(X,Y)),[1:p]'];
                   CO2 = sortrows(CO,1,'descend');
                   
                    % Calculate p matrix for train kernel matrix
                    MK = cell(p,1); 
                    for i = 1:p
                        AX = repmat(X(:,i),1,numTrain);
                        BX = AX';
                        lx = -sigma.*(AX-BX).^2;
                        MK{i} = exp(lx);
                    end
                    
                    % Calculate for test kernel matrix
                    TK = cell(p,1);
                    for i = 1:p
                        CX = repmat(test_data(:,i),1,numTrain);
                        DX = repmat(X(:,i)',numTest,1);
                        tx = -sigma.*(CX-DX).^2;
                        TK{i} = exp(tx);
                    end
                    
                   
            
                    k = 0; % iteration number
                    lambda = zeros (1,p);
                    nom = lambda;
                    ind = [CO2(:,2)']; %order of updating lambda
                    nv = [];
                    GG= [10000];
                    DD = [nom];
                    cc = 10000;
                    dd = 100;
                    
                     % Calculate kernel matrix K1
                    K1 = K2fun(MK,lambda,p);
                    
                    while(cc >= 0.001 || dd >= 0.001)
                         k =k+1;
                         if k > 500
                             Converflag = 0;
                             break
                         else
                             Converflag = 1;
                         end
                         
                         %calculate alpha_k
                         KH = K1 +gamma1*numTrain*eye(numTrain);
                         ee = min(eig(KH));
                         %if ee <= 10^(-4)
                         %    KH = KH + ee.*eye(numTrain);
                         %else
                             % do nothing
                         %end
                         
                             KH = KH + 10^(-4)*eye(numTrain);
                        
                         alpha = KH\Y;
                         alpha_k = alpha'; %row vector
                         ind(nv)=[];
                         if(isempty(ind))
                             break
                         end
                         
                         
                         
                         %updating lambda
                         for q = ind
                             %Calculate WM
                             WMT = 1+lambda(q).*MK{q};
                             WM = K1./WMT;
                             
                             %Calculate VM
                             VM = WM.*MK{q};
                             
                             %First calculate a_iq for each i and save in a vector:
                             A = Y - WM*alpha_k';
                             
                             % Calculate b_iq for each i and save in a vector:
                             B = -VM*alpha_k';
                             
                             % Calculate c_q for fixed q:
                             c_q = gamma1.*(alpha_k*VM*alpha_k');
                              
                             % Calculate d_q for fixed q:
                             d_q1 = c_q +gamma2;
                             
                             
                             %lambda(q) > theta(q)
                             v1 = -numTrain*d_q1 - 2*A'*B;
                             v = v1/(2*(B'*B));
                             h1 = min(max(v,0),10^5);
                                    
                             %Compare H1 and H2
                             lambda(q) = h1;
                             
                             %Update K1
                             UMM =1+lambda(q)*MK{q};
                             K1 = WM.*UMM;
                           
                         end
                                        
                         lambda;%after update lambda under a fixed alpha_k
                         gg = (norm(Y - K1*alpha_k'))^2/numTrain + gamma1*alpha_k*K1*alpha_k'...
                             +gamma2*sum(lambda(:));
                         GG = [GG;gg];
                         cc = abs(GG(k+1) -GG(k));
                         nv = find(lambda(:,ind)==0);
                         %nom = sum(lambda(:));
                         nom = lambda;
                         DD = [DD;nom];
                         %dd = abs(DD(k+1,:)-DD(k,:));
                         dd = sum(abs(DD(k+1,:)-DD(k,:)));
    
                    end
                    
                     % Calculate the Kernel matrix for the train data using new kernel
                     K1 = K2fun(MK,lambda,p);
                     alpha = (K1 +gamma1*numTrain*eye(numTrain))\(Y);
                     %updatekernel for test data
                     K2 = K2fun(TK,lambda,p);
                     CL =sqrt((norm(test_label- K2*alpha))^2/numTest);
                     
end
