function est_AMP = AMP(y,Cx,Cy,E,M,Ka,prior,compute_prior,nAMPiter,damping,update_prior,K,test)

% function beta_AMP = AMP(y,Cx,Cy,P,M,Ma,Ka,prior,prior_update,nAMPiter,damping,update_prior,k,test)
% 
% Function to run the AMP decoder described in Section III-A of 
%
%   [1] K.-H. Ngo, D. P. Krishnan, K. Okumus, G. Durisi, and E. G. Strom, 
%   "Type-based unsourced multiple access," arXiv preprint arXiv:2404.19552, Apr. 2024.
%
% written by Khac-Hoang Ngo, email: ngok@chalmers.se

N = length(y);          % blocklength
est_K = zeros(M,1);     % estimate of K across iterations, before rounding
prior = repmat(prior(:)',M,1);  % prior across M quantized positions

z = y;  % residual noise

est_AMP_old = est_K;

for idxIter = 1:nAMPiter
    % --- Compute tau using the residual
    tau = norm(z)/sqrt(N);

    % --- effective observation
    s = Cy(z) + sqrt(E)*est_K;

    % --- denoiser
    u = (s-(0:Ka)*sqrt(E)).^2;
    exptmp = exp((-u)/(2*tau^2)); 
    Z = sum(exptmp.*prior,2);
    W = sum(exptmp.*(prior.*(0:Ka)),2);
    est_K = (1-damping)*est_K + damping*W./Z;

    % --- terminating condition
    est_AMP = max(0,round(est_K));
    if sum(abs(est_AMP - est_AMP_old)) == 0
        if test
            fprintf('terminate at iteration %i\n',idxIter)
        end
        break
    end
    est_AMP_old = est_AMP;

    % --- residual
    dW = sum(( ((0:Ka)*sqrt(E) - s).*exptmp ).*(prior.*(0:Ka)),2)/tau^2;
    dZ = sum(( ((0:Ka)*sqrt(E) - s).*exptmp ).*prior,2)/tau^2;
    dF = sqrt(E)*(dW./Z - (dZ./Z).*est_K);

    z = y - sqrt(E)*Cx(est_K) + (M/N)*z*mean(dF);

    % --- update the prior
    if update_prior
        est_Ka = max(0,round(sum(est_K)));
        est_Ma = max(0,round(sum(est_AMP > 0)));
        prior = compute_prior(est_Ka,est_Ma,M);
        prior = repmat(prior(:)',M,1);
    end

    % --- check point
    if test
        figure(1)
        stem(K > 0)
        hold on 
        stem(est_AMP > 0,'*r:')
        xlim([1 M])
        legend('true','estimate')
        title('Sparsity indicator')
        hold off

        figure(2)
        stem(K)
        hold on 
        stem(est_AMP,'*r:') 
        hold off
        xlim([1 M])
        title('Type')
        legend('true','estimate')

        keyboard
    end
end
end
