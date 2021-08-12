%% Learning rules, grid simulation



clear all;


% choose one
% ACTION_SWITCH = 'RUNandSAVE';
ACTION_SWITCH = 'LOADandPLOT';
% ACTION_SWITCH = 'LOADandCOUNT';

% choose one
% RULE_SWITCH = 'SH'; % StandardHomeo
% RULE_SWITCH = 'CH'; % CrossHomeo
% RULE_SWITCH = 'TF'; % TwoFactor
RULE_SWITCH = 'GD'; % gradient-descent TwoFactor (with anti-homeo)



switch RULE_SWITCH
	case 'SH'
		data_filename = 'grid_standardHomeo_1.mat';
% 		data_filename = 'grid_standardHomeo_2.mat';
% 		data_filename = 'grid_standardHomeo_3.mat';
		n_steps = 2000;
	case 'CH'
		data_filename = 'grid_crossHomeo_1.mat';
		n_steps = 1000;
	case 'TF'
		data_filename = 'grid_twoFactor_1.mat';
% 		data_filename = 'grid_twoFactor_2.mat';
		n_steps = 1000;
	case 'GD'
		data_filename = 'grid_gradDescent_1.mat';
% 		data_filename = 'grid_gradDescent_2.mat';
		n_steps = 1000;
end
dt = 0.1;					% Time Step ms
t = dt*[1:n_steps];			% Time Array ms


%% Parameter values

g_E = 1;
g_I = 4;
E_set = 5;
I_set = 14;
Theta_E = 4.8;
Theta_I = 25;
tau_E = 10;
tau_I = 2;
alpha_EE = 0.02;
alpha_EI = 0.02;
alpha_IE = 0.02;
alpha_II = 0.02;
alpha = 0.02;%0.01;%0.02;
beta = 0.02;%0.05;%0.02;
beta_E = 0.01;
beta_I = 0.01;

params = cell2struct({g_E,g_I,E_set,I_set,Theta_E,Theta_I,tau_E,tau_I,alpha_EE,alpha_EI,alpha_IE,alpha_II,alpha,beta,beta_E,beta_I},...
		{'g_E','g_I','E_set','I_set','Theta_E','Theta_I','tau_E','tau_I','alpha_EE','alpha_EI','alpha_IE','alpha_II','alpha','beta','beta_E','beta_I'},2);


% % numerics
% n_steps = 2000;
% dt = 0.1;%2;%1;					% Time Step ms
% t = dt*[1:n_steps];			% Time Array ms



%% Theoretical relationships

% fixed point relationships - weights
W_EIup = @(W_EEup) ((E_set*W_EEup - Theta_E)*g_E - E_set)/(I_set*g_E);
W_IIup = @(W_IEup) ((E_set*W_IEup - Theta_I)*g_I - I_set)/(I_set*g_I);

% fixed point relationships -s activities
E_up = @(W_EE,W_EI,W_IE,W_II) (Theta_I*W_EI*g_I - (W_II*g_I + 1)*Theta_E)*g_E/((W_EI*W_IE*g_I - (W_II*g_I + 1)*W_EE)*g_E + W_II*g_I + 1);
I_up = @(W_EE,W_EI,W_IE,W_II) ((Theta_I*W_EE*g_I - Theta_E*W_IE*g_I)*g_E - Theta_I*g_I)/((W_EI*W_IE*g_I - (W_II*g_I + 1)*W_EE)*g_E + W_II*g_I + 1);
f_up = {E_up,I_up};

% stability conditions, neural subsystem
W_IEdetcond = @(W_EE) (Theta_I*W_EE*g_E - Theta_I)/(Theta_E*g_E);	% W_IE smaller than this value
W_IEtrcond = @(W_EE) (I_set*W_EE*g_E*tau_I + Theta_I*g_I*tau_E - I_set*tau_I)/(E_set*g_I*tau_E);	% W_IE greater than this value



if strcmp(ACTION_SWITCH,'RUNandSAVE')
%% Run: Initial conditions at full 4D grid

	W_EEinigrid = [1:1:10];
	W_EIinigrid = [0.5:0.5:5];
	W_IEinigrid = [2:2:20];
	W_IIinigrid = [1:1:10];
% 	W_EEinigrid = [1:1:4];
% 	W_EIinigrid = [0.5:0.5:2];
% 	W_IEinigrid = [2:2:8];
% 	W_IIinigrid = [1:1:4];
	W_inigrid = ndgrid(W_EEinigrid,W_EIinigrid,W_IEinigrid,W_IIinigrid);

	NEE = length(W_EEinigrid);
	NEI = length(W_EIinigrid);
	NIE = length(W_IEinigrid);
	NII = length(W_IIinigrid);
	W_grid = nan(NEE,NEI,NIE,NII,n_steps,4);
	for nee = 1:NEE
		W_EEini = W_EEinigrid(nee);
		for nei = 1:NEI
			W_EIini = W_EIinigrid(nei);
			for nie = 1:NIE
				W_IEini = W_IEinigrid(nie);
				for nii = 1:NII
					disp(['nee=' num2str(nee) '/' num2str(length(W_EEinigrid)) ', nei=' num2str(nei) '/' num2str(length(W_EIinigrid)) ', nie=' num2str(nie) '/' num2str(length(W_IEinigrid)) ', nii=' num2str(nii) '/' num2str(length(W_IIinigrid))]);
					W_IIini = W_IIinigrid(nii);
					W_ini = [W_EEini,W_EIini,W_IEini,W_IIini];
					W = [];
					switch RULE_SWITCH
						case 'SH'
							W = ode4(@(t,W) kernel_standardHomeo(t,W,f_up,params),t(1),dt,t(end),W_ini);
						case 'CH'
							W = ode4(@(t,W) kernel_crossHomeo(t,W,f_up,params),t(1),dt,t(end),W_ini);
						case 'TF'
							W = ode4(@(t,W) kernel_twoFactor(t,W,f_up,params),t(1),dt,t(end),W_ini);
						case 'GD'
							W = ode4(@(t,W) kernel_gradDescent(t,W,f_up,params),t(1),dt,t(end),W_ini);
					end
					W_grid(nee,nei,nie,nii,:,:) = W;
				end
			end
		end
	end

	save(data_filename,'W_grid','W_EEinigrid','W_EIinigrid','W_IEinigrid','W_IIinigrid','dt','params');


	

elseif strcmp(ACTION_SWITCH,'LOADandPLOT')
%% Load and plot full grid 3D

	load(data_filename);

	NEE = length(W_EEinigrid);
	NEI = length(W_EIinigrid);
	NIE = length(W_IEinigrid);
	NII = length(W_IIinigrid);

	W_EE_lims = [0 W_EEinigrid(end)];
	W_EI_lims = [0 W_EIinigrid(end)];
	W_IE_lims = [0 W_IEinigrid(end)];
	W_II_lims = [0 W_IIinigrid(end)];
	% choose one value of W_II
	W_II_idx = 1;


	fsize = 12;
	fsize_legend = 10;
	lwidth = 2;
	msize = 7;
	ptic1 = [];
	ptic2 = [];

	figure(4);
	clf(4);
	fig_size = [5 5];
	set(gcf,'PaperPositionMode','manual');
	set(gcf,'PaperSize',[fig_size],'PaperPosition',[0 0 fig_size]);

	skip_data = 1;	% skip datapoints from every run, 1=no skip
	skip_initial = 2;	% skip runs from the grid, 1=no skip
	for nee = 1:skip_initial:NEE
		for nei = 1:skip_initial:NEI
			for nie = 1:skip_initial:NIE
				for nii = W_II_idx
					W_EE = squeeze(W_grid(nee,nei,nie,nii,1:skip_data:end,1));
					W_EI = squeeze(W_grid(nee,nei,nie,nii,1:skip_data:end,2));
					W_IE = squeeze(W_grid(nee,nei,nie,nii,1:skip_data:end,3));
					W_II = squeeze(W_grid(nee,nei,nie,nii,1:skip_data:end,4));

					% workaround: prevent plotting outside of limits
					plot_limits = W_EE<0 | W_EE>W_EE_lims(2);
					W_EE(plot_limits) = [];
					W_EI(plot_limits) = [];
					W_IE(plot_limits) = [];
					W_II(plot_limits) = [];
					if ~isnan(W_EE)
						% basin of attraction
						plot3(W_EE,W_IE,W_EI,'g-','linewidth',lwidth);
						hold on;
						ptic1 = plot3(W_EE(1),W_IE(1),W_EI(1),'go','markersize',msize);
					else
						% diverging series
						plot3(W_EE,W_IE,W_EI,'k-','linewidth',lwidth);
						hold on;
						ptic2 = plot3(W_EE(1),W_IE(1),W_EI(1),'ko','markersize',msize);
					end
				end
			end
		end
	end

	% 2D plane attractor
	grid_step = 0.2;
	W_EEinigrid2 = [grid_step/2:grid_step/2:W_EEinigrid(end)];
	W_IEinigrid2 = [grid_step:grid_step:W_IEinigrid(end)];
	[x2,y2] = meshgrid(W_EEinigrid2,W_IEinigrid2);
	W_EIplane2 = W_EIup(x2);

	% intersection between plane attractor and stabilty conditions of the neural subsystem
	W_IEdetcond_x2 = W_EEinigrid2;
	W_IEdetcond_y2 = W_IEdetcond(W_EEinigrid2);
	W_IEtrcond_x2 = W_EEinigrid2;
	W_IEtrcond_y2 = W_IEtrcond(W_EEinigrid2);
	intersect_det_tr_idx = find(W_IEdetcond_y2 > W_IEtrcond_y2,1);

	% stable region of the plane attractor
	stable_region_1 = [W_EEinigrid2(intersect_det_tr_idx:end); W_IEdetcond(W_EEinigrid2(intersect_det_tr_idx:end))];
	stable_region_2 = [W_EEinigrid2(end); W_IEinigrid2(end)];
	stable_region_3 = fliplr([W_EEinigrid2(intersect_det_tr_idx:end); W_IEtrcond(W_EEinigrid2(intersect_det_tr_idx:end))]);
	stable_region = [stable_region_1, stable_region_2, stable_region_3]';
	[in,on] = inpolygon(x2(:),y2(:),stable_region(:,1),stable_region(:,2));
	mask = reshape(in|on,size(x2));
	W_EIplane2_stable = W_EIplane2;
	W_EIplane2_stable(~mask) = NaN;
	W_EIplane2_stable(W_EIplane2_stable<0) = NaN;
	% unstable region
	W_EIplane2_unstable = W_EIplane2;
	W_EIplane2_unstable(mask) = NaN;
	W_EIplane2_unstable(W_EIplane2_unstable<0) = NaN;

	pts1 = surf(x2,y2,W_EIplane2_stable);
	set(pts1,'facecolor',[0.5 0.75 0.5]);
	pts2 = surf(x2,y2,W_EIplane2_unstable);
	set(pts2,'facecolor',[0.95 0.95 0.95]);

	set(gca,'fontsize',fsize);
	if (~isempty(ptic1))
		if (~isempty(ptic2))
			ptl = legend([ptic1 ptic2],['Initial condition inside' char(10) 'basin of attraction'],['Initial condition outside' char(10) 'basin of attraction']);
		else
			ptl = legend([ptic1],['Initial condition inside' char(10) 'basin of attraction']);
		end
	else
			ptl = legend([ptic2],['Initial condition outside' char(10) 'basin of attraction']);
	end
	set(ptl,'location','northwest','fontsize',fsize_legend);
	title(['Initial value: WII_{ini}=' num2str(W_IIinigrid(W_II_idx))],'fontsize',fsize,'fontweight','bold');
	view([-20 25]);
% 	view([0 90]);
% 	view([48 19]);
	xlabel('W_{EE}');
	ylabel('W_{IE}');
	zlabel('W_{EI}');
	xlim(W_EE_lims);
	ylim(W_IE_lims);
	zlim(W_EI_lims);
	box on;

% 	print '-dpng' '-r150' 'crossHomeo_3Dview.png';
% 	print '-dpng' '-r150' 'crossHomeo_2Dview.png';
% 	print '-dpng' '-r150' 'twoFactor_3Dview.png';


elseif strcmp(ACTION_SWITCH,'LOADandCOUNT')
%% Load and estimate size of basin of attraction

	load(data_filename);

	NEE = length(W_EEinigrid);
	NEI = length(W_EIinigrid);
	NIE = length(W_IEinigrid);
	NII = length(W_IIinigrid);

	W_grid_end = squeeze(W_grid(:,:,:,:,end,:));
	W_grid_end_in = ~isnan(sum(W_grid_end,5));
	fraction_in_nan = sum(W_grid_end_in(:))/prod(size(W_grid_end_in));

	EI_end = zeros(NEE,NEI,NIE,NII);
	for nee = 1:NEE
		for nei = 1:NEI
			for nie = 1:NIE
				for nii = 1:NII
					WEE = W_grid_end(nee,nei,nie,nii,1);
					WEI = W_grid_end(nee,nei,nie,nii,2);
					WIE = W_grid_end(nee,nei,nie,nii,3);
					WII = W_grid_end(nee,nei,nie,nii,4);
					E_end = E_up(WEE,WEI,WIE,WII);
					I_end = I_up(WEE,WEI,WIE,WII);
					if (abs(E_end-E_set)<0.01*E_set && abs(I_end-I_set)<0.1*I_set)
						EI_end(nee,nei,nie,nii) = 1;
					end
				end
			end
		end
	end
	fraction_in_end = sum(EI_end(:))/prod(size(EI_end));


	disp([data_filename ', estimated size of basin of attraction:']);
	disp(['fraction_in_nan=' num2str(fraction_in_nan*100) '% of phase space']);
	disp(['fraction_in_end=' num2str(fraction_in_end*100) '% of phase space']);
	disp(' ');



end

%%
