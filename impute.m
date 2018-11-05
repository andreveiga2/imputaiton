function [ xx ] = impute( g_nr, phi, time, decision,  obs_r0, obs_r5, obs_r10, method  )


% all input vectors have the time dimension
% g_nr is a vector identifying the group of the individual (1,2,3, or 4)
% phi is a vector, its one of the predictor variables (welath)
% time is another predictor variable
% decision is which contract the individual bought
% obs_r0 is the observed price if the individual chose contract 0, and
% similar for 5 and 10
% method is the imputation method (random forest, etc)

% maybe we should train on the WHOLE data, g_nr, phi and time
% and then predict within each subset

% load('data/data_gar.mat');
GAR_levels = [ 0, 5, 10 ];

% convert missing values to NaN, only for visualizing 
obs_r0(obs_r0 == 9.99) = NaN;
obs_r5(obs_r5 == 9.99) = NaN;
obs_r10(obs_r10 == 9.99) = NaN;

% pre-allocate memory. these are the vectors that will be populated with
% the imputed rates
[r0, r5, r10] = deal(NaN(size(g_nr)));

Allvalues = [];

% loop through the 4 groups of consumers
% groups of consumers are indexed with i
% there are 4 consumer groups
for i=[1,2,3,4]
    
    disp(['Imputing group ' ns(i) ', method=' method])
    
    
    iii = (g_nr==i);
    % g_nr_i will have dimension Ni x 1 and have all entries = i
    % phi_i is the values of phi for individuals in group i
    [g_nr_i, phi_i, time_i] = deal(g_nr(iii), phi(iii), time(iii));
    
    % loop through contracts j
    for j=1:numel(GAR_levels)
        
        gar_j = GAR_levels(j);
        
        % g_nr_j = group number for individuals who buy j (from any group)
        %jjj = (decision==gar_j);
        %[g_nr_j, phi_j, time_j] = deal(g_nr(jjj), phi(jjj), time(jjj));
    
        % individuals from group i who buy j
        ijij = (g_nr==i & decision==gar_j); 
        [g_nr_ij, phi_ij, time_ij] = deal(g_nr(ijij), phi(ijij), time(ijij));
        
        
        % depending on the garantee under study, take a different vector of the "observed garantee"
        % there should be a more elegant way of doing this
%         if      gar_j==0;    r_ij = obs_r0(ijij);    rj = obs_r0(jjj);
%         elseif  gar_j==5;    r_ij = obs_r5(ijij);    rj = obs_r5(jjj);
%         elseif  gar_j==10;   r_ij = obs_r10(ijij);   rj = obs_r10(jjj);
%         end
%         
        
        if      gar_j==0;    r_ij = obs_r0(ijij);   
        elseif  gar_j==5;    r_ij = obs_r5(ijij);   
        elseif  gar_j==10;   r_ij = obs_r10(ijij);
        end
        
        % in any case, we always predict for group i, contract j
%         if 1==0
%             % here training data is everyone who bought contract j
%             % there is variation within g_nr_j
%             % this method requires a bigger number of trees
%             predictors = [g_nr_j, phi_j, time_j];
%             target = rj;
%             predict_at = [g_nr_i, phi_i, time_i];
%         else
%             % here training data is only individuals of group i who bough contract j
            % here there is NO VARIATION within g_nr_ij. ideally we'd take
            % them out but it's in here otherwise the code breaks
            % this seems to do better than the one above.
        predictors = [phi_ij, time_ij];                 % input for training
        target = r_ij;                                  % output for training
        predict_at = [phi_i, time_i];                   % input for prediction
        %end
        [ r_predicted_ij ] = impute_inner(predictors, target, predict_at, method );
                
        if ~isequal(size(r_predicted_ij), size(g_nr_i)); error('...'); end
        
        
        % replace the relevant rate of EVERYONE in group i with the estimated rates
        % depending on what contract we're looking at, save
        % output/prediction into the appropriate vector
        if gar_j==0;   r0(g_nr==i)      = r_predicted_ij;  end
        if gar_j==5;   r5( g_nr==i)     = r_predicted_ij;  end
        if gar_j==10;  r10( g_nr==i)    = r_predicted_ij; end
        
        
        % put everything into a structure
        hh.predictors = predictors;
        hh.target = target;
        hh.predict_at =  predict_at;
        
        hh.phi_i = phi_i;
        hh.time_i = time_i;
        hh.g_nr_i = g_nr_i;
        
        hh.g_nr_ij = g_nr_ij;
        hh.r_obs_ij = r_ij;                 % rates observed
        hh.phi_ij = phi_ij;
        hh.time_ij = time_ij;
        hh.r_predicted_ij = r_predicted_ij;
        hh.method = method;
        
        xx.group(i).gar(j) = hh;
    end
    
    % allocate predictions to the vectors r0, r5, r10
    xx.group(i).r0 = r0( g_nr==i  );
    xx.group(i).r5 = r5( g_nr==i  );
    xx.group(i).r10 = r10( g_nr==i  );
end

%csvwrite("file_b.csv", Allvalues);
return


%% build structure and save

disp(['Processing data into a structure...'])
alpha(:);                       % must do this first, otherwise it thinks alpha refers to the function alpha
data(4).alpha = NaN;            % pre-allocate structure
GROUPS = [1 2 3 4];             % groups to process
for i=1:numel(GROUPS)
    gi = GROUPS(i);
    % ok = (g_nr==gi & filter);           % filter
    ok = (g_nr==gi);           % filter
    
    % covariates
    data(gi).alpha           = alpha( ok );
    data(gi).phi             = phi( ok );
    data(gi).fa              = fa( ok );
    data(gi).int             = internal( ok );
    data(gi).LE              = LE( ok );
    data(gi).pcodeH          = dd_pcode_H( ok );
    data(gi).pcodeM          = dd_pcode_M( ok );
    
    % mortality data
    data(gi).entry           = buy_age( ok );
    data(gi).exit            = exit_age( ok );
    data(gi).died            = died( ok );
    
    % rates
    data(gi).r0             = r0( ok );
    data(gi).r5             = r5( ok );
    data(gi).r10            = r10( ok );
    
    % decisions
    data(gi).decisions      = decision( ok );
    data(gi).buy0           = data(gi).decisions == 0;
    data(gi).buy5           = data(gi).decisions == 5;
    data(gi).buy10          = data(gi).decisions == 10;
    
    % other stuff
    data(gi).N              = numel( data(gi).decisions );
    data(gi).name           = grp_name(gi);
    
end

save('data/data.mat','data')

%% some summaries of the data

disp(['max rate = ' ns(max(r0))  ', min rate = ' ns(min(r10))])
disp(['max LE = ' ns(max(LE)) ',  min LE=' ns(min(LE))])


return

