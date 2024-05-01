function est_EP = EP(y,C,Cx,Cy,E,M,Ka,prior,compute_prior,nIter,damping,update_prior,simplified,Vc,Dc,K,test)

% function est_EP = EP(y,C,Cx,Cy,P,M,Ma,Ka,prior,prior_update,nIter,
%               damping,update_prior,simplified,Vc,Dc,k,test)
% 
% Function to run the EP decoder described in Section III-B of 
%
%   [1] K.-H. Ngo, D. P. Krishnan, K. Okumus, G. Durisi, and E. G. Strom, 
%   "Type-based unsourced multiple access," arXiv preprint arXiv:2404.19552, Apr. 2024.
%
% written by Khac-Hoang Ngo, email: ngok@chalmers.se

N = length(y);          % blocklength
prior = repmat(prior(:)',M,1);  % prior across M quantized positions

%% Initialize the messages
mu0 = zeros(M,1);
xi0 = 1e6*ones(M,1);

mu1 = zeros(M,1);
xi1 = 1e6*ones(M,1);

est_EP_old = zeros(M,1);

%% EP iterations
for idxEP = 1:nIter
    % ----- update mu1, xi1
    u = (mu0-(0:Ka)).^2;
    exptmp = exp(-u./(2*xi0))./sqrt(xi0);
    pmf_tmp = prior.*exptmp;
    pmf_tmp = pmf_tmp./sum(pmf_tmp,2);

    mu_new = pmf_tmp*(0:Ka)';
    xi_new = sum(pmf_tmp.*(((0:Ka) - mu_new).^2),2);
    
    % if there is no more uncertainty, don't update
    idx_update = logical((xi_new > 0).*(xi0 > 0));
    if sum(idx_update) == 0
        if test
            fprintf('terminate at iteration %i\n',idxEP)
        end
        break
    end

    xi1(idx_update) = (1-damping)*xi1(idx_update) + ...
        damping./(1./xi_new(idx_update) - 1./xi0(idx_update));
    mu1(idx_update) = (1-damping)*mu1(idx_update) + ...
              damping*(mu_new(idx_update).*xi1(idx_update)./xi_new(idx_update) -...
                mu0(idx_update).*xi1(idx_update)./xi0(idx_update));

    % --- update the estimate
    if simplified
        xitmp = mean(xi1);
        xi1 = ones(M,1)*xitmp;
    end

    X1inv = diag(1./xi1);

    if simplified
        Xi0 = xitmp*(eye(M) - Vc./(1/E/xitmp./Dc' + 1)*Vc');
    else
        Xi1 = diag(xi1);
        Xi0 = Xi1 - Xi1*C'*((eye(N)/E + C*Xi1*C')\C)*Xi1;
    end
    xi0_new = diag(Xi0);
    mu0_new = Xi0*(X1inv*mu1 + Cy(y)*sqrt(E)); 

    % --- terminating condition
    est_EP = max(0,round(mu0_new));
    if sum(abs(est_EP - est_EP_old)) == 0
        if test
            fprintf('terminate at iteration %i\n',idxEP)
        end
        break
    end
    est_EP_old = est_EP;
    
    % --- update mu0, xi0
    xi0 = (1-damping)*xi0 + damping./(1./xi0_new - 1./xi1);
    mu0 = (1-damping)*mu0 + damping*max(0,xi0.*(mu0_new./xi0_new - mu1./xi1));
    mu0(xi0 == 0) = mu0_new(xi0 == 0);

    % --- update the prior
    if update_prior
        est_Ka = max(0,round(sum(est_EP)));
        est_Ma = max(0,round(sum(est_EP > 0)));
        prior = compute_prior(est_Ka,est_Ma,M);
        prior = repmat(prior(:)',M,1);
    end

    % --- check point
    if test
        figure(1)
        stem(K>0)
        hold on 
        stem(est_EP>0,'*r:')
        xlim([1 M])
        legend('true','estimate')
        title('Sparsity indicator')
        hold off
    
        figure(2)
        stem(K)
        hold on 
        stem(est_EP,'*r:')
        hold off
        xlim([1 M])
        title('Type')
        legend('true','estimate')
    
        keyboard
    end
end
