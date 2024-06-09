function [B,E,C,time] = JSSC_PAMcutSSC(X,beta,lambda,alph,rho,s,tol,max_iter)
%Solve the Joint Sparse Subspace Clustering by PAM algorithm
%
% min_{B,E,C} \frac{1}{2}\|X-XC-E\|_{F}^{2}+\lambda\|E\|_{2,1}~\\
%s.t. B=C,~~~\|B\|_{0,2}\le s,\\
%&C^{\top}\mathbf{1}=\mathbf{1},~\textrm{diag}(C)=\mathbf{0}.,
% 
% ---------------------------------------------
% Written by Yanjiao Zhu  (Zhuyanjiao258@163.com)
%
% initialization
obj=zeros(max_iter,1);
       MRes1=zeros(max_iter,1);
       CRes1=zeros(max_iter,1);
       Re =zeros(max_iter,1);
[D,N] = size(X);
affine = 0;
        alpha = 20;
        tic
        CMat = admmOutlier_mat_func(X, affine, alpha);
         C_SSC = CMat(1:N,:);
    E = zeros(D, N);
    C = C_SSC;  
    B =C_SSC;
    
    t0      = tic;

for iter = 1 : max_iter
    Bk =B;
    Ek = E;
    Ck = C;
%% update B by proj_l20
 P=(beta/(beta+rho))*Ck+(rho/(beta+rho))*Bk;
 B=proj_l20(P,s);
%% update E
 Tx    = find( sum(abs(X),1)>0 );
 XC    = X(:,Tx)*C(Tx,:);
       %E = solve_l1l2(X - XC + rho * Ek,2*lambda/(2+rho));
       
       E = prox_l12(X - XC + rho * Ek,lambda/(1+rho));
%% update C
Txx    = find( sum(abs(X),2)>0 );
 XX   = X(Txx,:)'*X(Txx,:);
    A = inv(XX+(beta+rho)*eye(N));  
    C = A*(XX-X(Txx,:)'*E(Txx,:)+beta*B+rho*Ck);
     C = C - diag(diag(C));
%% check convergence
    chgC    = norm(Ck(:)-C(:))/norm(Ck(:));
    
    if iter > 10
        if chgC < tol
            break;
        end
    end
     E21=0;
for e=1:size(E,2)
E21=E21+norm(E(:,e),2);
end
           obj(iter)=norm(X-X*C-E,'fro')^2+lambda*E21;
         if (iter-1)==0
             Re(iter)=obj(iter);
             MRes1(iter)=norm(B,'fro');
             CRes1(iter)=norm(C,'fro');
         else
         Re(iter)   =obj(iter)-obj(iter-1);
         MRes1(iter) =norm(B-Bk,'fro');
         CRes1(iter) =norm(C-Ck,'fro');
         end
         fprintf('%4d  %5.2e  %5.2e %5.2e %5.2e %5.2e \n',iter,obj(iter),Re(iter),MRes1(iter),CRes1(iter),norm(E-Ek,'fro')); 
        
    %disp(['iter ' num2str(iter) ',chgC=' num2str(chgC,'%2.1e')])
     time=toc(t0);
end

end