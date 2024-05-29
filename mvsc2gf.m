function [C_uv_post,Z_uv_post, C, Z, lambda, C_uv_residual, Z_uv_residual] = mvsc2gf(X, alpha, beta, eta)

  % min sum_i^v ||Yi - Ci*Yi||_F^2 + alpha*||Ci - FCi||_F^2 +(lambda_i)^eta* beta*||Ci -
  % C||_F^2                                                           
  % s.t. Yi = 3/4*Xi + 1/4*C*Xi
  %      Ci = Ci^t, Ci*e = e, Ci >= 0, diag(Ci) = 0
  %      C = C^t,   C*e = e,  C >= 0,  diag(C) = 0
  %      sum_i^v lambda_i = 1, lambda_i >= 0
  %                      |
  %                      |
  %                      V
  % min sum_i^v ||Yi - Ci*Yi||_F^2 + alpha*||Ci - FZi||_F^2 + beta*||Ci -
  % C||_F^2                                                           
  % s.t. Yi = 3/4*Xi + 1/4*C*Xi
  %      Ci = Zi, Zi = Zi^t, Zi >= 0, diag(Zi) = 0
  %      Ci*e = e, 
  %      C = Z,   C = C^t,   C >= 0,  diag(C) = 0 
  %      C*e = e
  %      sum_i^v lambda_i = 1, lambda_i >= 0


%% parameters

tol = 1e-4;
mu = 1e-4;
rho = 1.1;
max_mu = 1e30;
maxIter = 1500;

view_num = size(X, 2);
sample_num = size(X{1}, 1);

e = ones(sample_num,1);
I = eye(sample_num);
eet = e*e';

C_uv_post = zeros(sample_num, sample_num);
Z_uv_post = C_uv_post;



C_uv_residual = [];
Z_uv_residual = [];
%% auxiliary Variables and Lagrange multipliers

Y = cell(1, view_num);
Z = cell(1, view_num);
C = cell(1, view_num);
L1 = cell(1, view_num);
L2 = cell(1, view_num);
L3 = cell(1, view_num);
lambda = zeros(1, view_num);
for i=1:view_num
    C{i} = zeros(sample_num, sample_num);
    Z{i} = zeros(sample_num, sample_num);

    L1{i} = zeros(size(X{i}));
    L2{i} = zeros(sample_num, sample_num);
    L3{i} = zeros(sample_num, 1);

    lambda(i) = 1/view_num;
   
end

L_uv1 = zeros(sample_num, sample_num);
L_uv2 = zeros(sample_num, 1);

lambda_eta = compute_lambda_eta(lambda, eta, view_num);


%% Start main loop
iter  = 0;
while iter < maxIter
    iter = iter + 1;
    
    C_uv_pre = C_uv_post;
    Z_uv_pre = Z_uv_post;

    % CtC= C_uv_pre'*C_uv_pre;
    for i=1:view_num
    
        %% update Yi
        I_Ci = I - C{i};
        Ai = 2*(I_Ci'*I_Ci) + 16*mu*I;
        Bi = 12*mu*X{i} + 4*mu*C_uv_pre*X{i} - 4*L1{i};
        Y{i} = Ai\Bi;

        %% update Ci
        YYi = Y{i}*Y{i}';
        Ai = (2*YYi + 2*(alpha + lambda_eta(i)*beta)*I) + mu*(I + eet);
        Bi = (2*YYi + 2*alpha*C_uv_pre*Z{i} + 2*lambda_eta(i)*beta*C_uv_pre) + mu*(Z{i} + eet) - L2{i} - L3{i}*e';
        C{i} = Bi/Ai;
        
        %% update Zi

        Ai = 2*alpha*(C_uv_pre'*C_uv_pre)+ mu*I;
        Bi = 2*alpha*C_uv_pre'*C{i} + mu*C{i} + L2{i};
        Z{i} = Ai\Bi;
        
        Z{i} = 0.5*(Z{i} + Z{i}');
        Z{i} = Z{i} - diag(diag(Z{i}));
        Z{i} = max(0, Z{i});
        
    end

    
    %% update C
    A = zeros(sample_num, sample_num);
    B = A;
    for i=1:view_num
        ZZt = Z{i}*Z{i}';
        XXt = X{i}*X{i}';
        A = A + (2*alpha*ZZt + 2*lambda_eta(i)*beta*I) + mu*XXt;
        B = B + (2*alpha*C{i}*Z{i}' + 2*lambda_eta(i)*beta*C{i}) + 4*mu*Y{i}*X{i}' - 3*mu*XXt + L1{i}*X{i}';
    end
    A = mu*(I + eet) + A;
    B = B + mu*(Z_uv_pre + eet) - L_uv1 - L_uv2*e';
    C_uv_post = B/A;
    
    %% update Z
    Z_uv_post = C_uv_post + L_uv1/mu;
    % Z_uv_post = 0.5*(Z_uv_post + Z_uv_post');
    Z_uv_post = Z_uv_post - diag(diag(Z_uv_post));
    Z_uv_post = max(Z_uv_post, 0);

    lambda = compute_obj(C, C_uv_post, eta, view_num);
    lambda_eta = compute_lambda_eta(lambda, eta, view_num);

    stopC = [];
    for i=1:view_num
        leq_mv1{i} = 4*Y{i} - 3*X{i} - 1*(C_uv_post*X{i});    
        leq_mv2{i} = C{i} - Z{i};
        leq_mv3{i} = C{i}*e - e;
        stopC = [stopC, max([max(max(abs(leq_mv1{i}))),max(max(abs(leq_mv2{i}))), max(max(abs(leq_mv3{i})))])];
    end
    
    leq_uv1 = C_uv_post - Z_uv_post;
    leq_uv2 = C_uv_post*e - e;
    
    stopC = max([stopC, max(max(abs(leq_uv1))), max(max(abs(leq_uv2)))]);
    
    C_uv_residual = [C_uv_residual, norm(C_uv_post-C_uv_pre,'fro')];
    Z_uv_residual = [Z_uv_residual, norm(Z_uv_post - Z_uv_pre,'fro')];

    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(' ')
        disp(['iter=' num2str(iter) ', mu=' num2str(mu,'%2.1e') ...
            ', leq_uv1=' num2str(max(max(abs(leq_uv1)))) ...
            ', leq_uv2=' num2str(max(max(abs(leq_uv2))))])
        for i=1:view_num
            disp(['leq_mv1=' num2str(max(max(abs(leq_mv1{i})))) ...
                  ', leq_mv2=' num2str(max(max(abs(leq_mv2{i})))) ...
                  ', leq_mv3=' num2str(max(max(abs(leq_mv3{i})))) ])
        end
        
    end
    if stopC<tol 
        break;
    else
        L_uv1 = L_uv1 + mu*leq_uv1;
        L_uv2 = L_uv2 + mu*leq_uv2;

        for i=1:view_num
            L1{i} = L1{i} + mu*leq_mv1{i};
            L2{i} = L2{i} + mu*leq_mv2{i};
            L3{i} = L3{i} + mu*leq_mv3{i};
        end
        mu = min(max_mu,mu*rho);
    end


end


function lambda_eta = compute_lambda_eta(lambda, r,view_num)
lambda_eta = zeros(1, view_num);
for i=1:view_num
    lambda_eta(i) = lambda(i)^r;
end

function lambda = compute_obj(C, C_uv, r, view_num)
obj = zeros(1, view_num);
lambda = zeros(1, view_num);
for i=1:view_num
    obj(i) = norm(C{i} - C_uv,"fro");
end
obj_r = obj.^(1/(1-r));
for i=1:view_num
    lambda(i) = obj_r(i)/sum(obj_r);
end
