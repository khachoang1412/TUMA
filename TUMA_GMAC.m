function data = TUMA_GMAC(N,Ka,B_list,Ma_list,SNRdB_list,area_size,codebook_type,...
        method,perfect_comm,update_prior,nMC,nIter,test)

% function data = TUMA_GMAC(n,Ka,B_list,Ma_list,SNRdB_list,area_size,codebook_type,...
%       method, perfect_comm,update_prior,nMC,nIter,test)
%
% This function is a simulation of the multi-target position tracking
% scenario via type-based unsourced multiple access (TUMA) considered in:
%
%   [1] K.-H. Ngo, D. P. Krishnan, K. Okumus, G. Durisi, and E. G. Strom, 
%   "Type-based unsourced multiple access," April 2024. 
%
% We consider a scenario where Ka sensors track the position of Ma targets 
% placed uniformly at random in a square area S. The area is divided into a 
% regular grid consisting of M = 2^B disjoint cells whose centroids form 
% the set of quantized positions Q. 
% 
% Each sensor tracks a target chosen uniformly at random from the Ma 
% targets. Sensor k determines the position of its target, maps this 
% position to the closest quantized position, and report the index of this
% quantized position to the receiver using a communication codebook C.
% 
% The receiver aims to aims to estimate the type of the positions, i.e., 
% the set of distinct positions and their multiplicity in the sequence of
% quantized positions reported by all sensors. For this, the receiver uses 
% one of the following decoder: 
%   1. approximate message passing (AMP)
%   2. scalar AMP
%   3. expectation propagation (EP).
% See [1, Section III] for a description of these decoders. We also 
% consider a simplified version of EP, which is not reported in [1].
%
% INPUTS:
%   N       : blocklength
%   Ka      : number of sensors 
%   B_list  : quantization resolution, number of bits to represent a
%            quantized position
%   Ma_list : number of targets
%   SNRdB_list    : transmit signal to noise ratio (SNR) in dB
%   area_size     : length of each side of S
%   codebook_type : how the communication codebook is contructed, either
%                   'Gaussian' or 'Hadamard'
%   method  : decoding method, 'AMP' or 'EP' or 'simplifiedEP' or 'scalarAMP'
%   perfect_comm  : set this to 1 to let the decoding be error-free, and 0
%                   otherwise
%   update_prior  : set this to 1 to update the prior along the iterations,
%                   and 0 otherwise
%   nMC     : number of Monte-Carlo iterations
%   nIter   : number of decoding iterations
%   test    : set this to 1 to enable some checkpoints to test the decoder,
%               and 0 to run the whole simulation without stopping
%
% OUTPUTS:
%   data : a structure that stores all the inputs, as well as the
%          communication performance (total variation distance) and overall
%          performance (Wasserstein distance).
%
% written by Khac-Hoang Ngo, email: ngok@chalmers.se

%%
DEBUG = 1;

if DEBUG
    close all

    N = 500;                % codeword length
    Ka = 50;                % number of sensors
    B_list = 10;            % number of bits per codeword
    Ma_list = [Ka:-5:5];    % number of active codewords
    area_size = 1;          % length of each side of the square area in metter

    SNRdB_list = -12;       % transmit SNR

    perfect_comm = 0;   
    update_prior = 0;
    codebook_type = 'Hadamard'; % 'Gaussian' or 'Hadamard'

    nMC = 1e2;              % number of Monte-Carlo iterations
    nIter = 10;             % number of decoding iterations
    
    method = 'AMP'; % 'AMP' or 'EP' or 'simplifiedEP' or 'scalarAMP'
    
    test = 0;
end

% function to compute the prior
f_prior = @(Ka,Ma,M) compute_prior(Ka,Ma,M);

% damping does not help, so we set it to 1 (no damping)
damping = 1;    

%% indicator if a simplified version of EP is used
simplifiedEP = 0;
if strcmpi(method,'simplifiedEP')
    simplifiedEP = 1;
end

%% TV distance
TV_distance = @(p,q) 0.5*sum(abs(p-q));

%% Initialization
SNR = db2pow(SNRdB_list);

TV = zeros(length(SNR),length(B_list),length(Ma_list)); % total variation
WS = zeros(length(SNR),length(B_list),length(Ma_list)); % Wasserstein

TV_ML = zeros(length(SNR),length(B_list),length(Ma_list)); 
        % a (probably loose) lower bound on the TV achieved with ML decoding

%% Simulation
for idxMa = 1:length(Ma_list)
    Ma = Ma_list(idxMa);
    
    for idxB = 1:length(B_list)
        B = B_list(idxB);
        M = 2^B;  % number of quantized positions
        
        % Average combining gain due to multiple sensors reporting the same target
        combining_gain = avg_combining_gain(Ka,Ma,M); 

        % Compute the prior according to [1, Eq. (6)]
        % If Ka and Ma are unknown, they can be replaced by initial values
        prior = f_prior(Ka,Ma,M);
        
        % Generate the codebook C and define functions to compute C*x and C'*y
        C = 0;
        if N >= M   % orthogonal codebook possible
            C = eye(N);
            C = C(:,1:M);
            
            Cx = @(x) C*x;          
            Cy = @(y) C'*y;
        elseif strcmpi(codebook_type,'Gaussian')
            C = randn(N,M);
            C = C./vecnorm(C);
            
            Cx = @(x) C*x;
            Cy = @(y) C'*y;
        elseif strcmpi(codebook_type,'Hadamard')
            if N > M
                ordering = randsample(N,M);
                C = fwht(eye(N))*N/sqrt(N);
                C = C(:,ordering);

                Cx = @(x) C*x;
                Cy = @(y) C'*y;
            else
                ordering = randsample(M,N);
                Cx = @(x) sub_fwht(x,M,N,ordering);
                Cy = @(y) sub_ifwht(y,M,N,ordering);
            
                if strcmpi(method,'EP') || strcmpi(method,'simplifiedEP') ...
                        || strcmpi(method,'scalarAMP')
                    C = fwht(eye(M))*M/sqrt(N);
                    C = C(ordering,:);
                end
            end
        end
        
        % Precompute some quantities that will be repeatedly used by the decoder
        if strcmpi(method,'EP') || strcmpi(method,'simplifiedEP')
            [~,Dc,Vc] = svd(C);
            Vc = Vc(:,1:N);
            Dc = diag(Dc).^2;
        else
            Vc = 0;
            Dc = 0;
        end

        if strcmpi(method,'scalarAMP')
            C2 = C.^2;
        else
            C2 = 0;
        end
        
        for idxSNR = 1:length(SNR)
            tic
            E = N*SNR(idxSNR); % energy of a codeword

            fprintf('\n-------\n')
            fprintf(['Ma = %i objects,  B = %i bits, txSNR = %1.3f dB, ' ...
                'rxEbN0 = %2.3f dB \n\n'],Ma,B,pow2db(E/N),pow2db(combining_gain*E/2/B))
        
            TV_tmp = 0;
            WS_tmp = 0;
    
            TV_ML_tmp = 0;
        
            for idxMC = 1:nMC % use parfor to parallelize
                % ----- generate Ma positions
                positions = area_size*(-1/2 + rand(Ma,1) +1i*(-1/2 + rand(Ma,1)));
        
                % ----- message selection
                active_msg = qamdemod(positions*2*ceil(sqrt(M))/area_size,M) + 1;
                quantized_positions = area_size*qammod(active_msg-1,M)/2/ceil(sqrt(M));
            
                % ----- sensor selection
                tx_type = mnrnd(Ka,ones(Ma,1)/Ma)';
                
                % ----- number of sensors reporting each quantized position
                K = zeros(M,1);
                for ii = 1:length(active_msg)
                    K(active_msg(ii)) = K(active_msg(ii)) + tx_type(ii);
                end
    
                % ----- received signal
                y = sqrt(E)*Cx(K) + randn(N,1);
                
                % ----- detection
                if perfect_comm     % the estimation of k is perfect
                    est_K = K;
                else
                    if strcmpi(method,'AMP')
                        est_K = AMP(y,Cx,Cy,E,M,Ka,prior,f_prior,nIter,damping,update_prior,K,test);
                    elseif strcmpi(method,'EP') || strcmpi(method,'simplifiedEP')
                        est_K = EP(y,C,Cx,Cy,E,M,Ka,prior,f_prior,nIter,damping,update_prior,simplifiedEP,Vc,Dc,K,test);
                    elseif strcmpi(method,'scalarAMP')
                        est_K = scalarAMP(y,C2,Cx,Cy,E,M,Ka,prior,f_prior,nIter,damping,update_prior,K,test);
                    end
                end
    
                % ---- if the output is nonsense, give a random guess
                if sum(est_K) == 0
                    guess_active_msg = randsample(M,Ma);
                    est_K = zeros(M,1);
                    est_K(guess_active_msg) = mnrnd(Ka,ones(Ma,1)/Ma)';
                end
                
                % ----- TV distance
                TV_tmp = TV_tmp + TV_distance(K/sum(K), est_K/sum(est_K));

                % ----- lower bound on ML performance
                est_K_ML = K;
                if norm(y - sqrt(E)*Cx(K)) > norm(y - sqrt(E)*Cx(est_K))
                    est_K_ML(1:2) = est_K_ML(1:2) + [0,1]';
                end
                TV_ML_tmp = TV_ML_tmp + TV_distance(K/sum(K), est_K_ML/sum(est_K_ML));
        
                % ----- rx type and Wasserstein distance
                rx_type = est_K(est_K > 0);
                [x,ws_distance] = emd(positions, area_size*qammod(find(est_K > 0)-1,M)/2/ceil(sqrt(M)), ...
                    tx_type/sum(tx_type), rx_type/sum(rx_type), @gdf, 2);
                WS_tmp = WS_tmp + ws_distance;
            end
        
        % update the performance metrics
        TV(idxSNR,idxB,idxMa) = TV_tmp/nMC;
        WS(idxSNR,idxB,idxMa) = WS_tmp/nMC;

        TV_ML(idxSNR,idxB,idxMa) = TV_ML_tmp/nMC;
        
        fprintf('Total variation:      %1.3e \n', TV(idxSNR,idxB,idxMa))
        fprintf('Wasserstein distance: %1.3e \n \n', WS(idxSNR,idxB,idxMa))

        fprintf('ML Total variation (lower bound):      %1.3e \n\n', TV_ML(idxSNR,idxB,idxMa))
        toc
        end
    end
end

%% Store the results
data.method = method;
data.n = N;
data.Ka = Ka;
data.B = B_list;
data.Ma = Ma_list;
data.SNR_dB_list = SNRdB_list;
data.side = area_size;
data.codebook_type = codebook_type;
data.perfect_comm = perfect_comm;
data.update_prior = update_prior;
data.nMC = nMC;
data.nIter = nIter;
data.damping = damping;

data.TV = TV;
data.WS = WS;
data.TV_ML = TV_ML;

if ~DEBUG
    %% Save the results
    filename = ['TUMA_GMAC_' method '_n_' num2str(N) '_Ka_' num2str(Ka) ...
        '_B_' num2str(min(B_list)) 'to' num2str(max(B_list)) ...
        '_Ma_' num2str(min(Ma_list)) 'to' num2str(max(Ma_list)) ...
        '_SNR_' num2str(min(SNRdB_list)) 'to' num2str(max(SNRdB_list)) ...
        'dB_' codebook_type '.mat'];
    save(filename, 'data', '-v7.3');
else
    %% Plot TV and WS
    % ---- vs SNR
    if length(B_list) == 1 && length(Ma_list) == 1
        figure(1)
        semilogy(SNRdB_list,TV)
%         hold on
%         semilogy(SNR_dB_list,TV_ML,'--')
%         legend('AMP','ML')
        xlabel('SNR (dB)')
        ylabel('Total variation distance')
        
        figure(2)
        semilogy(SNRdB_list,WS)
        xlabel('SNR (dB)')
        ylabel('Wasserstein distance')
    end
    
    % --- vs B
    if length(SNRdB_list) == 1 && length(Ma_list) == 1
        figure(1)
        semilogy(B_list,TV)
%         hold on
%         semilogy(B_list,TV_ML,'--')
%         legend('AMP','ML')
        xlabel('number of bits')
        ylabel('Total variation distance')
        
        figure(2)
        semilogy(B_list,WS)
        xlabel('number of bits')
        ylabel('Wasserstein distance')
    end

    % --- vs Ma
    if length(SNRdB_list) == 1 && length(B_list) == 1
        figure(1)
        semilogy(Ma_list,squeeze(TV))
%         hold on
%         semilogy(Ma_list,squeeze(TV_ML),'--')
%         legend('AMP','ML')
        xlabel('number of objects')
        ylabel('Total variation distance')
        
        figure(2)
        semilogy(Ma_list,squeeze(WS))
        xlabel('number of objects')
        ylabel('Wasserstein distance')
    end

    keyboard
end
end

%% Function to compute the prior according to [1, Eq.(6)]
function prior = compute_prior(Ka,Ma,M)
    pmf_ni = binopdf(0:Ma,Ma,1/M);
    prior = zeros(Ka+1,1);
    for ni = 0:Ma
        prior = prior + pmf_ni(ni+1)*binopdf(0:Ka,Ka,ni/Ma)';
    end
end

%% Function to compute the average combining gain
function gain = avg_combining_gain(Ka,Ma,M)
    pmf_ni = binopdf(1:Ma,Ma,1/M);
    pmf_ni = pmf_ni/sum(pmf_ni);

    pmf_Ki = binopdf(0:Ka,Ka,1/Ma);

    grid = (1:Ma)'.*(0:Ka);
    pmf_grid = pmf_ni'.*pmf_Ki;

    gain = sum(grid.^2.*pmf_grid,'all');
end

%% Functions to compute C*x and C'*y for the Hadarmard codebook
function Cx = sub_fwht(x,M,n,ordering)
    Cx = fwht(x)*M/sqrt(n);
    Cx = Cx(ordering,:);
end

function Cy = sub_ifwht(y,M,n,ordering)
    z = zeros(M,size(y,2));
    z(ordering,:) = y;
    Cy = ifwht(z)/sqrt(n);
end