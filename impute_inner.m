function [ r_predicted_ij ] = impute_inner(predictors, target, predict_at, method)


% phi_i, time_i, phi_ij, time_ij, r_ij_obs,

% things with ..._obs are data used as inputs into the imputation
% things with ..._to_impute are the values of phi and time at which
% r_predicted_ij is to be evaluated

% phi_ij_obs = observed wealth (phi) for group i who chose contract j 
% r_ij_obs = observed rates for group i who chose contract j 
% time_ij_obs = observed time period for group i who chose contract j

% [phi_i, time_i] = pairs of [phi, time] at which rates must be imputed

% --------
% THINGS TO DO:
% try splines

% try r = g(phi) + h(t), ie assume phi and t independent, estimated g(.)
% and h(.) by, say, randomforest

% is there a way to "smooth" the random forest imputation. It's quite
% jagged in phi currently
% --------

%%


nr_trees = 1;         % number of trees in RF


%% random forest with continuous t as input

% 1) train model on group i, choice j
% 2) predict for entire group i
% monotonicity is better for 1000 trees than 10 trees, but not much better
% this is currently the best and also the most flexible model
% it's also very easy to code
if isequal(method, 'RF')
    [r,~] = size(predictors); % Step 1
    randomRowIdxs = randperm(r); % Step 2
    shuffled_predictors = predictors(randomRowIdxs,:);
    shuffled_target = target(randomRowIdxs);
    trained_model_ij = TreeBagger(nr_trees, shuffled_predictors, shuffled_target, 'Method', 'regression', 'NumPredictorsToSample','all');
    r_predicted_ij = predict(trained_model_ij, predict_at);

    
end

% view(trained_model_ij.Trees{1},'mode','graph')


%% function to create dummy variables

% the native matlab function doesnt work well, it creates dummies even for
% values which do not occur in the input vector

function [dummies] = make_dummies(input_vec)
    % produces a matrix of dummy variables
    % automatically excludes the first one, so if you want to include that
    % (say, for prediction), you need to include a vector of all zeros
    % ie, if input_vec has X unique elements, then dummies will have
    % (X-1) columns
    N = numel(input_vec);
    list = unique(input_vec);            % list of unique time points
    T = numel(list);                   % number of time periods in input data
    dummies = NaN(N, T);        % pre allocate memory
    for z = 1:T
        dummies(:,z) = (input_vec == list(z) );
    end
    dummies = dummies(:,2:end);           % Normalise the first time period to 0 (Eliminate first period, otherwise dummies will be multicoliner)
end











%% shallow neural net with continuous t as input

% code based on https://uk.mathworks.com/help/deeplearning/gs/fit-data-with-a-neural-network.html
% single hidden layer with some number of neurons


if isequal(method, 'NN')
    inputs = predictors;
    targets = target;
    
    % Create a Fitting Network
    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

    hiddenLayerSize1 = 12;
    hiddenLayerSize2 = 12;
    net = fitnet([hiddenLayerSize1,hiddenLayerSize2],trainFcn);
    net.trainParam.epochs=5; %more epochs

    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};

    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivide
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 10/100;

    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean Squared Error

%    %Training Parameters
%    %net.trainParam.show=1;  %# of ephocs in display
%     %net.trainParam.epochs=10000;  %max epochs
% 
%         % Choose Plot Functions
%         % For a list of all plot functions type: help nnplot
%         net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
%             'plotregression', 'plotfit'};
% 
%         % Train the Network
          % Test the Network
    
    % Train the Network
    [net,tr] = train(net,inputs,targets);
    
    % Test the Network
    r_predicted_ij = net(predict_at);
end






%% random forest with dummy variables as inputs

% 1) train model on group i, choice j
% 2) predict for entire group i
% there is a problem because sometimes the training data does not include a
% particular time period.
if isequal(method, 'RF-dummy')
    timme_dummies_ij = dummyvar(time_ij);
    predictors_ij = [phi_ij, timme_dummies_ij];
    trained_model_ij = TreeBagger(nr_trees, predictors_ij, r_ij_obs, 'Method', 'regression');
    timme_dummies_i = dummyvar(time_i);
    predictors_i = [phi_i, timme_dummies_i];
    r_predicted_ij = predict(trained_model_ij, predictors_i);
end








%% random forest with fixed effects

% uses r = g(phi) + time_FE, and g(.) is estimated by random forest

% these is an issue with missing time periods within a given ij
% nr_T might be < max(time_ij_obs)
% but if we included a column in t_dummies for each time period, things
% wouldn't work because at some point there will be a non-invertible matrix

if isequal(method, 'RF-FE')
    N_ij = numel(phi_ij);               % number of observations in i-j
    t_list = unique(time_ij);            % list of unique time points
    T_ij = numel(t_list);                   % number of time periods in input data
    
    % generate dummy variables for group i, contract j
    t_dummies = make_dummies(time_ij);

    % 1) regress phi on r
    trained_model = TreeBagger(nr_trees, phi_ij, r_ij_obs,'Method','regression');
    ghat_rate = predict(trained_model, phi_ij);
    ehat_rate = r_ij_obs - ghat_rate;       % compute the residual after the predicted rates from regression of phi on r
    
    % 2) regress  phi on time dummies
    [ ghat_time, ehat_time] = deal( NaN(N_ij, T_ij-1) );
    for t = 1:size(t_dummies,2)
        trained_model = TreeBagger(nr_trees, phi_ij, t_dummies(:,t),'Method','regression');
        ghat_time(:,t) = predict(trained_model, phi_ij);
        ehat_time(:,t) = t_dummies(:,t) - ghat_time(:,t);
    end
    
    % (3) Estimates of time dummy effects by OLS on the NW regression residuals
    % bhat are the coefficients on the time fixed effects
    bhat = (ehat_time'*ehat_time)\(ehat_time'*ehat_rate);
    t_FE = [0;bhat];   % Time effects are the values of the bhat, preceded by a zero value for the first period
    
    % (4) Performing NW regression on the rate residuals after accounting for time fixed effects
    rate_bhat = r_ij_obs - t_dummies*bhat;           % What is left from rate after taking the estimated time effects
    
    % generating predicted rates
    M = TreeBagger(nr_trees,phi_ij,rate_bhat,'Method','regression');
    g_funds = predict(M, phi_i);
    
    % output (requires generate dummy variables for all of group i)
    % there is an issue here, if 
    N_i = numel(phi_i);
    t_dummies_i = NaN(N_i, T_ij);        % pre allocate memory for dummy variables
    % here it's important that we only generate nr_T dummies, since we have
    % only estimated nr_T dummies
        for t = 1:T_ij
            t_dummies_i(:,t) = (time_i == t_list(t) );
        end
    r_predicted_ij = g_funds + t_dummies_i*t_FE;  
end
% clf; plot(t_FE); pause(10)







%% using non-parametric regression with fixed effects
% uses r = g(phi) + time_FE, and g(.) is estimated by non parametric
% regression

% the bandwiths can affect the result quite a lot
% in particular the phi_bandwith needs to be quite high
[t_kernel, phi_kernel]  = deal('Gaussian');
[t_bandwidth, phi_bandwidth] = deal(200, 800);
    
if isequal(method, 'NPR-FE')
     %N_ij = numel(phi_ij);
    t_list = unique(time_ij);            % time_ob is the input of everyone's time observation; time_list are just the unique values
    T_ij = numel(t_list);               % number of time periods in input data
    
    % generate dummy variables
    t_dummies = make_dummies(time_ij);

    % (1) NW regression of rates on funds
    ghat_rate = kreg(phi_ij,r_ij_obs,phi_ij,phi_bandwidth,phi_kernel);
    ehat_rate = r_ij_obs - ghat_rate;
    
    % (2) NW regression of time dummies on funds
    % bandwith must be large enough
    ghat_time = NaN(size(t_dummies));
    for t = 1:size(t_dummies,2)
        ghat_time(:,t) = kreg(phi_ij, t_dummies(:,t), phi_ij, t_bandwidth, t_kernel);
        ehat_time(:,t) = t_dummies(:,t) - ghat_time(:,t);
    end
    
    % (3) Estimates of time dummy effects by OLS on the NW regression residuals
    bhat = (ehat_time'*ehat_time)\(ehat_time'*ehat_rate);
    t_FE = [0;bhat];                % Time effects are the values of the bhat, preceded by a zero value for the first period
    
    % (4) Performing NW regression on the rate residuals after accounting for
    % time fixed effects
    rate_bhat = r_ij_obs - t_dummies*bhat;           % What is left from rate after taking the estimated time effects
    
    % Regression of funds_ob on rate_bhat (rate left after time effects)
    g_funds = kreg(phi_ij,rate_bhat,phi_i,phi_bandwidth,phi_kernel);
    N_i = numel(phi_i);
    t_dummies_i = NaN(N_i, T_ij);        % pre allocate memory for dummy variables
    for t = 1:T_ij
        t_dummies_i(:,t) = (time_i == t_list(t) );
    end
    r_predicted_ij = g_funds + t_dummies_i*t_FE;
end





%% using splies


if isequal(method, 'spline')
    % Chaitanya, please fill this in    
    size(predictors)
    size(target)
    size(predict_at)
    r_predicted_ij = interpn(predictors,target,predict_at,'cubic', -1);
end





%% make output monotonic

% this makes the predicted rats monotonic, but its very brute forcy and
% messes up everything else. 
% r_predicted_ij = lsqisotonic(phi_i, r_predicted_ij, []); 


end






