function [ ] = impute_graph( inputs_hh  )



disp('Graphing imputation results...')

% this should produce graphic output of the imputation for a pretty generic
% set of imputation methods
% inside this routine, there is some additional imputation. This is because
% we might want to get general patterns for the imputed function, even at
% points at which we don't have to imput. So in this routine, typically we
% impute the necessary functions at more points that will be necessary for
% estimation


% the 3 levels of possible guarantees
GAR_levels = [ 0, 5, 10 ];


% rows is the number os rows of the output plot of graphs
% cols is the number of cols
% pp will change across graphs, in this case pp=-2 will make all graphs
% fall into place given the loops
[rows,cols] = deal(4,5); clf;
pp = -cols+1;

% loop through the 4 groups of consumers
% groups of consumers are indexed with i
% there are 4 consumer groups
for i=[1,2,3,4]
    
    group_name = grp_name(i);
    pp = pp + cols; % this starts the plot counter at the start of every row
    
    for j =1:numel(GAR_levels)
        
        hh_ij = inputs_hh.group(i).gar(j);
        
        predictors = hh_ij.predictors;
        target = hh_ij.target;
        
        g_nr_i          = hh_ij.g_nr_i;
        phi_i           = hh_ij.phi_i;
        time_i          = hh_ij.time_i;
        
        g_nr_ij         = hh_ij.g_nr_ij;
        r_obs_ij        = hh_ij.r_obs_ij;           % rate observed in the data (input for imputation)
        phi_ij          = hh_ij.phi_ij;
        time_ij         = hh_ij.time_ij;
        r_predicted_ij   = hh_ij.r_predicted_ij;
        method = hh_ij.method;                      % method used for imputation
        
        
        
        % in-sample fit (scatter of observed vs predicted for training data)
        % for the last contract (j=3), add legend and the 45 degree line
        % first run the imputation algorithm to predict rates those rates
        % which are observed
        predict_at = [g_nr_ij, phi_ij, time_ij];
        if size(predictors,2)==2;  predict_at(:,1) = []; end            % make sure they're same size
        r_predicted_and_obs = impute_inner(predictors, target, predict_at, method);
        subplot(rows, cols, pp);
        scatter(r_obs_ij, r_predicted_and_obs, 'o' );
        if j==3; legend(make_leg('g=',[0,5,10]),'Location','Best'); end
        if j==3; x = linspace(0.04, 0.09); line(x,x); end               % plot the 45 degree line only once
        err = 100*(r_obs_ij - r_predicted_and_obs)./r_obs_ij;
        err = mean(abs(err));
        title([group_name ', E[|error|]=' ns(err) '%']);
        hold on;
        
        
        
        
        % density of errors
        subplot(rows, cols, pp+1);
        [f,xi] = ksdensity(err); plot(xi,f);
        if j==3; legend(make_leg('g=',[0,5,10]),'Location','Best'); end
        xlabel('|error|')
        title([group_name ', |error| density']);
        hold on;
        
        
        
        % plot all predicted rates, for each contract
        % r over phi, for a fixed t
        J = 200;                                            % number of simulated indiviuduals/values of phi
        max_phi = max(phi_i);                                   % max value of phi for this group
        simm_phi_i = linspace(0, max_phi, J)';               % range of values of phi
        simm_t_i = ones(J,1)*round(mean(time_i));           % for the output everyone has the same time period: assign average of time_i
        % r_pred_ij_fixed_t = impute_inner(simm_phi_i, simm_t_i, phi_ij, time_ij, r_obs_ij, method );
        if var(g_nr_i)>0; error('...'); end                 % check we are dealing with a single group
        oo = ones(J,1)*mean(g_nr_i);
        predict_at = [oo, simm_phi_i, simm_t_i];
        if size(predictors,2)==2;  predict_at(:,1) = []; end 
        r_pred_ij_fixed_t = impute_inner(predictors, target, predict_at, method);
        subplot(rows,cols,pp+2);
        plot(simm_phi_i, r_pred_ij_fixed_t);
        if j==3; legend(make_leg('g=',[0,5,10]),'Location','Best'); end
        title([group_name ', r(\phi), fixed t']);
        xlabel('\phi');
        hold on
        
        
        
        % average level of rates over time
        nr_T = 24;              % time periods go 1,2,...,24
        tt = 1:nr_T;            % vector of time periods
        av_r = NaN(1,nr_T);         % allocate memory for average r for every time period
        J = 100;
        % the following few lines generate a long vector uniformely
        % distributed between grid_min and grid_max
        nplot = [J, nr_T];                              % density of points for each variable (phi and t)
        grid_min = [min(phi_i),  min(tt)];                      % upper bounds
        grid_max = [max(phi_i),   max(tt)];                      % lower bounds
        x_grid = nodeunif(nplot, grid_min, grid_max);
        [simm_phi_i, simm_t_i] =  deal_cols(x_grid);
        % r_pred_ij = impute_inner(simm_phi_i, simm_t_i, phi_ij, time_ij, r_obs_ij, method );    % predict rates for group i, contract j
        oo = ones(size(simm_phi_i))*mean(g_nr_i);
        predict_at = [oo, simm_phi_i, simm_t_i];
        if size(predictors,2)==2;  predict_at(:,1) = []; end 
        r_pred_ij = impute_inner(predictors, target, predict_at, method);    % predict rates for group i, contract j
        % loop through time period, compute average rate paid for this
        % contract over time periods
        for k=1:nr_T
           av_r(k) = mean( r_pred_ij( simm_t_i == tt(k)) );
        end
        subplot(rows, cols, pp+3);
        plot(tt, av_r)
        if j==3; legend(make_leg('g=',[0,5,10]),'Location','Best'); end
        title(['E_{\phi}[r] over time'])
        xlabel('t');
        hold on
    end
    
    % monotonicity: a major measure of how well we're doing
    % it should be r0 > r5 > r10
    % below plots a density of r0-r5 and r5-r10
    group_name = grp_name(i);
    r0 = inputs_hh.group(i).r0;
    r5 = inputs_hh.group(i).r5;
    r10 = inputs_hh.group(i).r10;
    rates_non_monotonic = (r0<r5 | r5<r10 | r0<r10 );
    pct_non_monotonic = 100*mean(rates_non_monotonic);
    subplot(rows, cols, pp+4);
    [f,xi] = ksdensity(r0-r5); plot(xi,f); hold on;
    [f,xi] = ksdensity(r5-r10); plot(xi,f); hold on;
    [f,xi] = ksdensity(r0-r10); plot(xi,f); hold on;
    legend('r0-r5','r5-r10','r0-r10','Location','Best')
    title([group_name ', non-monotonic=' ns(pct_non_monotonic) ' %']);
end

suptitle(['Imputing r(\phi,t), method=' method])
mysave( gcf, strcat('price_impute/'), strcat('Imputed_',method), 2*[rows,cols]);

disp('Graphing input results: done')

return






%% 






%% make small graph for paper

clf
grp_nr = 1;             % choose which group to graph
[rows,cols,pp] = deal(1,2,0);
clf

for g= [1,5,10]
    subplot(rows,cols,1);
    plot( ss(grp_nr).gar(g).sorted , ss(grp_nr).gar(g).g_fund ); hold on
    xlabel('\phi')
    title('r(\phi)')
    legend(make_leg('g=',[0,5,10]),'Location','Best')
    hold on;
end
suptitle([grp_name(grp_nr) ])
mysave( gcf, strcat('price_impute/trees/'), strcat('Imputed_eg'), 2);



return

