function [ cal ] = gen_cal( )


%% root path

root_path = '/Users/andreveiga/Desktop/Dropbox/PAPERS/JMP/matlab/annuities';

%% calibrated parameters and contract options

RR = 1.0313;            % average interest rate over this period;
cal.T = 100;                 % number of periods; below 65 introduces numerical instability for low alpha
cal.delta = 1/RR;           % discount rate
cal.gamma = 2;              % CRRA risk aversion parameter
cal.eta = 4;                % initial wealth (beyond annuity) is phi*eta
cal.lambda = 0.11;          % lambda shape parameter in Gompertz mortality
cal.kappa = 0;

%% all contracts 

inflation_year = 0.02311891;
cal.inflation = 1/(1+inflation_year);                   % calibration inflation during this period
cal.gL = 1*cal.inflation;
cal.gH = 1.03*cal.inflation;
cal.g = cal.gL;
cal.R = RR;

%% VF options

cal.vf.method = 'FOC';               % method for computing VF
cal.vf.grid = 5;                           % grid size
cal.vf.tol = 10^-11;                      % level of tolerance when computing VF (only used if method is 'VFc')
cal.vf.delta_c = 300;
cal.vf.delta_c_reduction = (1/3);

%% VF interpolation options

cal.max_beta = 100000;                     % maximum value of beta to user in interpolation
cal.min_beta = 0;
cal.min_alpha = alpha_from_LE(35, cal.lambda);
cal.max_alpha = 0.01;
% cal.min_alpha = 4e-5;
% cal.max_alpha = 1e-3;
cal.max_r = 0.085;
cal.min_r = 0.050;
cal.intrp_graphs = 1;

%% partial CR options

cal.CR.points = 10;            % number of intermediate CR points to evaluate

%% method for finding eql

% cal.eql.method = 'max-local'
cal.eql.method = 'max';                % use iteration to find equilibrium rates
cal.eql.method = 'iter';                % use iteration to find equilibrium rates

%% computing equilibrium by maximization

% no longer being used
% cal.eql.max.KK = [5e5, 50, 5];
% this will depend a lot on the number of replicas
cal.eql.max.KK = [1e5, 20, 3];                      
cal.eql.max.local_KK_fac = 1/5;                 % reduction in number of trial points when doing local equilibrium approx
cal.eql.max.ub = 0.15 * ones(1,3);              % highghest rate 
cal.eql.max.lb = 0.06 * ones(1,3);
cal.eql.max.min_val = -10;
cal.eql.graphs = 1;                     % produce graphs after finding eql (by any method)

%% computing eql by iteration

cal.eql.iter.iterations = 5000;               % number of iterations
cal.eql.iter.tol = 1;
cal.eql.iter.graphs = 1;                % produce graphs after finding eql by iteration
cal.eql.iter.coeff      =  1*(1e-9);
% cal.eql.iter_coeff    =  1*(1e-7);

%% computing beta* thresholds

% this method is currently not being used
cal.beta_star.KK = [200,1,1];
cal.beta_star.method = 'fminbdd';

%% ML options

cal.ML.smooth = 1e6;
cal.ML.KK =  [1e5, 20, 10];
cal.ML.beta_nodes = 15;                         % must be >5
cal.ML.quad_method = 'gauss';
% cal.ML.quad_method = 'unif';
cal.ML.opts = optimset('Display', 'off','Algorithm','sqp','TolX',1e-5,'TolFun',1e-5,'MaxIter',1e4, 'MaxFunEvals', 1e5);            % rigorous fmincon options

cal.ML.distribution = 'Lognormal';
cal.ML.distribution = 'Gauss';


cal.ML.method = 'util';
% cal.ML.method = 'thresh';

%% GLL (mortality GLL) options

cal.GLL.quad_nodes = 10;                  % GLL quadrature nodes - this could possibly make a big difference to estimate the variance
    
%% default model

cal.model = [];

%% rate imputation

cal.impute.nr_trees = 500;
cal.impute.use_time_dummies = 0;

%% typical individual

cal.e.phi = 12000;                  % this affects the scale of utilities. Must be large enough for numerical stability (20 is too low).
cal.e.LE = 24;
cal.e.alpha = alpha_from_LE(cal.e.LE, cal.lambda);
cal.v.alpha = 0.01;
cal.v.beta = 0.5;
cal.e.beta = 100;
cal.e.r = 0.060;
cal.e.rH = 0.040;
cal.e.rL = 0.060;
cal.e.R = 1/cal.delta;
cal.e.zeta5 = 1.001;
cal.e.gar = 5;
cal.e.w1 = 15000;
cal.e.lambda = cal.lambda;
cal.e.g = cal.g;
cal.e.growth = cal.g;

%% what warning messages to display

cal.silent.LL_check = 1;
cal.silent.choice_probs = 0;
cal.silent.VF = 1;
cal.silent.LL = 0;
cal.silent.w_equiv = 1;

%% market simulation

% draws from the distribution of beta, for each individual in sample; 
% makes a big difference to speed in computing the eql
% more replicas seems to make equilibrium harder to find
% also, this increases CPU time linearly
cal.replicas = 3;                  

%% whether to compute CS during market_outcomes

% this is turned off while finding eql, to make things faster
cal.compute_CS = 1;

%% other options

cal.root_path = root_path;          % path where to save outputs

%% save

save('temp/cal.mat','cal');          % save structure of calibrated parameters
    
end

