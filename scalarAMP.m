function est_scalarAMP = scalarAMP(y,C2,Cx,Cy,E,M,Ka,prior, ...
    compute_prior,nIter,damping,update_prior,K,test)

% function est_scalarAMP = scalarAMP(y,C2,Cx,Cy,P,M,Ma,Ka,prior,...
%               prior_update,nIter,damping,update_prior,k,test)
% 
% Function to run the scalar decoder described in Section III-C of 
%
%   [1] K.-H. Ngo, D. P. Krishnan, K. Okumus, G. Durisi, and E. G. Strom, 
%   "Type-based unsourced multiple access," April 2024. 
%
% written by Khac-Hoang Ngo, email: ngok@chalmers.se

N = length(y);          % blocklength
prior = repmat(prior(:)',M,1);  % prior across M quantized positions

% scale the output for convenience
noiseVar = 1/E;
y = y*sqrt(noiseVar); 

%% Initialization
Z = y;
V = ones(N,1);
est_mean = zeros(M,1);
est_var = ones(M,1);

est_scalarAMP_old = zeros(M,1);

for idxIter = 1:nIter
    % 1.a. DECOUPLING STEP
    % --- update V
    V_old = V;
    V = C2*est_var; 

    % --- update Z
    Z_old = Z;
    Z = Cx(est_mean) - V.*(y - Z)./(noiseVar + V_old);

    % --- update phi
    xi = 1./(C2'*((noiseVar+V).^(-1)));

    % --- update r
    r = est_mean + xi.*(Cy((y-Z)./(noiseVar + V)));

    % 1.b. DAMPLING STEP
    V = (1-damping)*V_old + damping*V;
    Z = (1-damping)*Z_old + damping*Z;

    % 1.c. DENOISING STEP
    % --- update est_mean
    u = (r-(0:Ka)).^2;
    exptmp = exp((-u)./(2*xi))./sqrt(xi);
    Pr = sum(exptmp.*prior,2);
    est_mean = sum(exptmp.*(prior.*(0:Ka)),2)./Pr;
    est_scalarAMP = max(0,round(est_mean));
    
    % --- termination condition
    if sum(abs(est_scalarAMP - est_scalarAMP_old)) == 0
        if test
            fprintf('terminate at iteration %i\n',idxIter)
        end
        break
    end
    est_scalarAMP_old = est_scalarAMP;

    % --- update est_var
    est_var = sum(exptmp.*(prior.*((0:Ka).^2)),2)./Pr - est_mean.^2;
    
    % --- update prior
    if update_prior
        est_Ka = max(0,round(sum(est_scalarAMP)));
        est_Ma = max(0,round(sum(est_scalarAMP > 0)));
        prior = compute_prior(est_Ka,est_Ma,M);
        prior = repmat(prior(:)',M,1);
    end
    
    % --- check point
    if test
        figure(1)
        stem(K>0)
        hold on 
        stem(est_scalarAMP > 0,'*r:')
        xlim([1 M])
        legend('true','estimate')
        title('Sparsity indicator')
        hold off
    
        figure(2)
        stem(K)
        hold on 
        stem(est_scalarAMP,'*r:') % 
        hold off
        xlim([1 M])
        title('Type')
        legend('true','estimate')
    
        keyboard
    end
end
end