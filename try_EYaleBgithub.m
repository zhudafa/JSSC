clc;
clear;

load('data/PIE_N1340_D1024.mat');
addpath('L1_ADMM/');
addpath('toolbox/');
addpath('LRR_code\');
addpath('LSR\');
addpath('LHCF\');
addpath('JSSC\');
addpath('AGCSC\');
addpath('NSKC\');
addpath('AMGSSC\');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Data processing
dataSetName = 'PIE';
nExperiment = 1;
data_num = size(X, 1);
class_num = length(unique(Y));
label = Y;  
nCluster = length(unique(label));



K=[4,5];
beta=[0.025];
lambda = [0.1,1];
alph = [0.0001];
 rho =[0.01]; 
corruption = 0;%add noise
corruption1 = [0];%¡–∆∆À
maxiter=linspace(1,6,6);
%maxiter=5;

results_JSSCSSC=zeros(nExperiment, 4);


 for cc=1:length(corruption1)
    corruption1_cc = corruption1(cc);
for kk=1:length(K)
    K_kk = K(kk);
for bb=1:length(beta)
    beta_bb=beta(bb);
for ll=1:length(lambda)
    lambda_ll=lambda(ll);        
for aa=1:length(alph)
     alph_aa=alph(aa); 
for rr=1:length(rho)
     rho_rr=rho(rr);
for ii=1:length(maxiter)
    maxiter_ii=maxiter(ii);
    for iExperiment = 1:nExperiment
        
        %add noise
    [D,N] = size(X);
a=zeros(size(X));
corruption_mask = randperm( D*N, round( corruption*D*N ) );
a(corruption_mask)=X(corruption_mask);
%a = imnoise(a,'speckle',0.01);%≥À–‘‘Î“Ù
%a = imnoise(a,'salt & pepper',0.01);%Ω∑—Œ‘Î“Ù
%a = imnoise(a,'poisson');%≤¥À…‘Î
a=imnoise(a,'gaussian',0,0.01);%∏ﬂÀπ‘Î“Ù
X(corruption_mask)=a(corruption_mask);
corruption_mask1=randperm(N);%¡–∆∆À
b=fix(corruption1_cc*N);
X(:,corruption_mask1(1:b))=zeros(D,b);
     
        X = NormalizeFea(X, 1);    %%% Normalization
   
        s=label;
  Z=X'\X';
     figure(1)

 colorbar
map = [1 1 1    %∂®“Â—’…´±‰¡ø
    1 0.94118 0.96078
    1 0.89412 0.88235
    1 0.62745 0.47843
    1 0.49804 0.31373
    1 0.27059 0];
 clims = [0 0.5];
 A=imagesc(abs(Z)',clims);
 
 %A=imagesc(abs(Z));
 colorbar
colormap(map)
    %JSSC
    tol = 1e-6;
    normalizeColumn = @(data) cnormalize_inplace(data);
  %JSSCSSC
    %tol = 1e-6;
    %normalizeColumn = @(data) cnormalize_inplace(data);
   % [B,E,C_JSSC,time_JSSC] = JSSC_PAMnoalpha(X',beta_bb,lambda_ll,0,rho_rr,K_kk,tol,maxiter);
[BSSC,ESSC,C_JSSCSSC,time_JSSCSSC] = JSSC_PAMcutSSC(X',beta_bb,lambda_ll,0,rho_rr,K_kk,tol,maxiter_ii);
 %C_JSSC = C_JSSC(1:N,:);
 C_JSSCSSC = C_JSSCSSC(1:size(X,1),:);
         W_JSSCSSC = abs(C_JSSCSSC) + abs(C_JSSCSSC');   %%%  affinity matrix
         Y_JSSCSSC = SpectralClustering1(W_JSSCSSC, nCluster);
  %EvaluationJSSC   
        accr_JSSCSSC  = evalAccuracy(s, Y_JSSCSSC);
        nmi_JSSCSSC  = nmi(Y_JSSCSSC, s);
       dataformat_JSSCSSC = '%d-th experiment:  accr_JSSC = %f, nmi_JSSC = %f, time=%f\n';
       dataValue_JSSCSSC = [iExperiment, accr_JSSCSSC,nmi_JSSCSSC, time_JSSCSSC]; 
       results_JSSCSSC(iExperiment, :)=dataValue_JSSCSSC; 
 
       
end

% output
 dataValue_JSSCSSC=mean(results_JSSCSSC, 1);

 
 
 fprintf('\nAverage:corruption=%d,iter=%d,beta=%d,lambda=%d,alpha=%d,rho=%d,K=%d:  accr_JSSCSSC = %f,nmi_JSSCSSC = %f, time_JSSCSSC=%f\n', corruption1(cc),maxiter(ii),beta(bb),lambda(ll),alph(aa),rho(rr),K(kk),dataValue_JSSCSSC(2:end));

end
   end
end
end
end
end
 end