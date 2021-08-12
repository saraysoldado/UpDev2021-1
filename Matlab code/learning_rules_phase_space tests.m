%% Learning rules, grid simulation



clear all;



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
% alpha_IE = 0.001;
% alpha_II = 0.001;
alpha = 0.02;
beta = 0.02;
beta_E = 0.01;
beta_I = 0.01;

params = cell2struct({g_E,g_I,E_set,I_set,Theta_E,Theta_I,tau_E,tau_I,alpha_EE,alpha_EI,alpha_IE,alpha_II,alpha,beta,beta_E,beta_I},...
		{'g_E','g_I','E_set','I_set','Theta_E','Theta_I','tau_E','tau_I','alpha_EE','alpha_EI','alpha_IE','alpha_II','alpha','beta','beta_E','beta_I'},2);


% numerics
n_steps = 1000;%1000;
dt = 0.1;%2;%1;					% Time Step ms
t = dt*[1:n_steps];			% Time Array ms



%% Theoretical relationships

% fixed point relationships - weights
W_EIup = @(W_EEup) ((E_set*W_EEup - Theta_E)*g_E - E_set)/(I_set*g_E);
W_IIup = @(W_IEup) ((E_set*W_IEup - Theta_I)*g_I - I_set)/(I_set*g_I);

% fixed point relationships, activities
E_up = @(W_EE,W_EI,W_IE,W_II) (Theta_I*W_EI*g_I - (W_II*g_I + 1)*Theta_E)*g_E/((W_EI*W_IE*g_I - (W_II*g_I + 1)*W_EE)*g_E + W_II*g_I + 1);
I_up = @(W_EE,W_EI,W_IE,W_II) ((Theta_I*W_EE*g_I - Theta_E*W_IE*g_I)*g_E - Theta_I*g_I)/((W_EI*W_IE*g_I - (W_II*g_I + 1)*W_EE)*g_E + W_II*g_I + 1);
f_up = {E_up,I_up};

% stability conditions, neural subsystem
W_IEdetcond = @(W_EE) (Theta_I*W_EE*g_E - Theta_I)/(Theta_E*g_E);	% W_IE smaller than this value
W_IEtrcond = @(W_EE) (I_set*W_EE*g_E*tau_I + Theta_I*g_I*tau_E - I_set*tau_I)/(E_set*g_I*tau_E);	% W_IE greater than this value




%% Single run

% % initial conditions at fixed point
% W_EEini = 10;
% W_IEini = 6;
% W_ini = [W_EEini,W_EIup(W_EEini),W_IEini,W_IIup(W_IEini)];

% arbitrary initial conditions
W_EEini = 5;
W_EIini = 10;
W_IEini = 5;
W_IIini = 2;
W_ini = [W_EEini,W_EIini,W_IEini,W_IIini];

% W = ode4(@(t,W) kernel_standardHomeo(t,W,f_up,params),t(1),dt,t(end),W_ini);
% W = ode4(@(t,W) kernel_crossHomeo(t,W,f_up,params),t(1),dt,t(end),W_ini);
% W = ode4(@(t,W) kernel_twoFactor(t,W,f_up,params),t(1),dt,t(end),W_ini);
W = ode4(@(t,W) kernel_gradDescent(t,W,f_up,params),t(1),dt,t(end),W_ini);


disp(['E=' num2str(E_up(W(end,1),W(end,2),W(end,3),W(end,4))) ', E_set=' num2str(E_set)]);
disp(['I=' num2str(I_up(W(end,1),W(end,2),W(end,3),W(end,4))) ', I_set=' num2str(I_set)]);

figure(1);
clf(1);

plot(t,W,'-')



%% GRID: Initial conditions at the fixed point

W_EEinigrid = [1:1:10];
W_EIinigrid = 0.99*W_EIup(W_EEinigrid);
W_IEinigrid = [2:2:20];
W_IIinigrid = 0.99*W_IIup(W_IEinigrid);
W_inigrid = ndgrid(W_EEinigrid,W_EIinigrid,W_IEinigrid,W_IIinigrid);

NEE = length(W_EEinigrid);
NEI = length(W_EIinigrid);
NIE = length(W_IEinigrid);
NII = length(W_IIinigrid);
W_grid = nan(NEE,NIE,n_steps,4);

for nee = 1:NEE
	disp(['nee=' num2str(nee) '/' num2str(length(W_EEinigrid))]);
	W_EEini = W_EEinigrid(nee);
	W_EIini = W_EIinigrid(nee);
	for nie = 1:NIE
		disp(['    nie=' num2str(nie) '/' num2str(length(W_IEinigrid))]);
		W_IEini = W_IEinigrid(nie);
		W_IIini = W_IIinigrid(nie);
		W_ini = [W_EEini,W_EIini,W_IEini,W_IIini];
		W = [];
% 		[t2,W] = ode45(@(t,W) kernel_standardHomeo(t,W,f_up),t,W_ini,options);
		W = ode4(@(t,W) kernel_crossHomeo(t,W,f_up),t(1),dt,t(end),W_ini);
% 		[t2,W] = ode15s(@(t,W) kernel_crossHomeo(t,W,f_up),t,W_ini,options);
% 		[t2,W] = ode45(@(t,W) kernel_crossHomeo(t,W,f_up),t,W_ini,options);
		W_grid(nee,nie,:,:) = W;
	end
end



%% Plot stable region


W_EE_lims = [0,10.5];
W_IE_lims = [0,21];

detcond_x = [W_EE_lims(1):1:W_EE_lims(2)];
detcond_y = W_IEdetcond(detcond_x);
trcond_x = [W_EE_lims(1):1:W_EE_lims(2)];
trcond_y = W_IEtrcond(detcond_x);



figure(2);
clf(2);

plot(detcond_x,detcond_y,'g--');
hold on;
plot(trcond_x,trcond_y,'g-');

for nee = 1:NEE
	for nie = 1:NIE
		W_EE = squeeze(W_grid(nee,nie,:,1));
		W_IE = squeeze(W_grid(nee,nie,:,3));

% 		if find(~isnan(W_EE),1,'first') && isempty(find(W_EE<=0,1,'first'))
		if ~isnan(W_EE)
			plot(W_EE,W_IE,'k-');
			plot(W_EE(1),W_IE(1),'k.');
% 			disp([num2str(nee) ',' num2str(nie)]);
		end
	end
end

xlim(W_EE_lims);
ylim(W_IE_lims);



%% Run grid: Initial conditions at full 4D grid

W_EEinigrid = [1:1:10];
W_EIinigrid = [0.5:0.5:5];
W_IEinigrid = [2:2:20];
W_IIinigrid = [1:1:10];
W_inigrid = ndgrid(W_EEinigrid,W_EIinigrid,W_IEinigrid,W_IIinigrid);

NEE = length(W_EEinigrid);
NEI = length(W_EIinigrid);
NIE = length(W_IEinigrid);
NII = length(W_IIinigrid);
W_grid = nan(NEE,NEI,NIE,NII,n_steps,4);
for nee = 1:NEE
	disp(['nee=' num2str(nee) '/' num2str(length(W_EEinigrid))]);
	W_EEini = W_EEinigrid(nee);
	for nei = 1:NEI
		disp(['.    nei=' num2str(nei) '/' num2str(length(W_EIinigrid))]);
		W_EIini = W_EIinigrid(nei);
		for nie = 1:NIE
			disp(['.    .    nie=' num2str(nie) '/' num2str(length(W_IEinigrid))]);
			W_IEini = W_IEinigrid(nie);
			for nii = 1:NII
				disp(['.    .    .    nii=' num2str(nii) '/' num2str(length(W_IIinigrid))]);
				W_IIini = W_IIinigrid(nii);
				W_ini = [W_EEini,W_EIini,W_IEini,W_IIini];
				W = [];
				% [t2,W] = ode45(@(t,W) kernel_standardHomeo(t,W,f_up),t,W_ini,options);
% 				[t2,W] = ode45(@(t,W) kernel_crossHomeo(t,W,f_up),t,W_ini,options);
% 				W = ode4(@(t,W) kernel_crossHomeo(t,W,f_up),t(1),dt,t(end),W_ini);
% 				W = ode4(@(t,W) kernel_standardHomeo(t,W,f_up),t(1),dt,t(end),W_ini);
				W = ode4(@(t,W) kernel_twoFactor(t,W,f_up),t(1),dt,t(end),W_ini);
				W_grid(nee,nei,nie,nii,:,:) = W;
			end
		end
	end
end


% save('grid_CrossHomeo_2.mat','W_grid','W_EEinigrid','W_EIinigrid','W_IEinigrid','W_IIinigrid','dt');
% save('grid_StandardHomeo_2.mat','W_grid','W_EEinigrid','W_EIinigrid','W_IEinigrid','W_IIinigrid','dt');
save('grid_twoFactor_2.mat','W_grid','W_EEinigrid','W_EIinigrid','W_IEinigrid','W_IIinigrid','dt');



%% PCA


W_pca = [];
for nee = 1:2:NEE
	disp(['nee=' num2str(nee) '/' num2str(length(W_EEinigrid))]);
% 	W_EEini = W_EEinigrid(nee);
	for nei = 1:2:NEI
		disp(['.    nei=' num2str(nei) '/' num2str(length(W_EIinigrid))]);
% 		W_EIini = W_EIinigrid(nei);
		for nie = 1:2:NIE
% 			disp(['.    .    nie=' num2str(nie) '/' num2str(length(W_IEinigrid))]);
% 			W_IEini = W_IEinigrid(nie);
			for nii = 1:2:NII
% 				disp(['.    .    .    nii=' num2str(nii) '/' num2str(length(W_IIinigrid))]);
% 				W_IIini = W_IIinigrid(nii);
% 				W_ini = [W_EEini,W_EIini,W_IEini,W_IIini];
% 				W = [];
				% [t2,W] = ode45(@(t,W) kernel_standardHomeo(t,W,f_up),t,W_ini,options);
% 				[t2,W] = ode45(@(t,W) kernel_crossHomeo(t,W,f_up),t,W_ini,options);
% 				W = ode4(@(t,W) kernel_crossHomeo(t,W,f_up),t(1),dt,t(end),W_ini);
				W_pca = [W_pca; squeeze(W_grid(nee,nei,nie,nii,:,:))];
			end
		end
	end
end

%%

coeff = pca(W_pca);
W_grid_pca = nan(NEE,NEI,NIE,NII,n_steps,4);


for nee = 1:2:NEE
	disp(['nee=' num2str(nee) '/' num2str(length(W_EEinigrid))]);
	for nei = 1:2:NEI
		disp(['.    nei=' num2str(nei) '/' num2str(length(W_EIinigrid))]);
		for nie = 1:2:NIE
% 			disp(['.    .    nie=' num2str(nie) '/' num2str(length(W_IEinigrid))]);
			for nii = 1:2:NII
% 				disp(['.    .    .    nii=' num2str(nii) '/' num2str(length(W_IIinigrid))]);
				W_grid_pca(nee,nei,nie,nii,:,:) = squeeze(W_grid(nee,nei,nie,nii,:,:))*coeff;
			end
		end
	end
end



%% Plot full grid 2D


% load('grid_CrossHomeo_1.mat');
% load('grid_CrossHomeo_2.mat');

% W_EEinigrid = [1:1:10];
% W_EIinigrid = [1:1:10];
% W_IEinigrid = [1:1:10];
% W_IIinigrid = [1:1:10];
% W_inigrid = ndgrid(W_EEinigrid,W_EIinigrid,W_IEinigrid,W_IIinigrid);

NEE = length(W_EEinigrid);
NEI = length(W_EIinigrid);
NIE = length(W_IEinigrid);
NII = length(W_IIinigrid);

W_EE_lims = [0 W_EEinigrid(end)];
W_EI_lims = [0 W_EIinigrid(end)];
W_IE_lims = [0 W_IEinigrid(end)];
W_II_lims = [0 W_IIinigrid(end)];

fixed_point_WEI_x = [W_EE_lims(1):1:W_EE_lims(2)]; 
fixed_point_WEI_y = W_EIup(fixed_point_WEI_x);
fixed_point_WII_x = [W_IE_lims(1):1:W_IE_lims(2)]; 
fixed_point_WII_y = W_IIup(fixed_point_WII_x);


figure(3);
clf(3);

subplot(1,2,1);
hold on;
subplot(1,2,2);
hold on;

skip = 20;
for nee = 1:2:NEE
	for nei = 1:2:NEI
		for nie = 1:2:NIE
			for nii = 1:2:NII
				W_EE = squeeze(W_grid(nee,nei,nie,nii,1:skip:end,1));
				W_EI = squeeze(W_grid(nee,nei,nie,nii,1:skip:end,2));
				W_IE = squeeze(W_grid(nee,nei,nie,nii,1:skip:end,3));
				W_II = squeeze(W_grid(nee,nei,nie,nii,1:skip:end,4));

				subplot(1,2,1);
				plot(fixed_point_WEI_x,fixed_point_WEI_y,'k--');
				plot(W_EE,W_EI,'m-');
				
				subplot(1,2,2);
				plot(fixed_point_WII_x,fixed_point_WII_y,'k--');
				plot(W_IE,W_II,'m-');
			end
		end
	end
end

subplot(1,2,1);
xlim(W_EE_lims);
ylim(W_EI_lims);

subplot(1,2,2);
xlim(W_IE_lims);
ylim(W_II_lims);


%% Plot full grid 3D

W_EE_lims = [0 W_EEinigrid(end)];
W_EI_lims = [0 W_EIinigrid(end)];
W_IE_lims = [0 W_IEinigrid(end)];
W_II_lims = [0 W_IIinigrid(end)];


lwidth = 2;
msize = 10;


figure(4);
clf(4)


skip = 1;
for nee = 1:2:NEE
	for nei = 1:2:NEI
		for nie = 1:2:NIE
			for nii = 1%1:2:NII
				W_EE = squeeze(W_grid(nee,nei,nie,nii,1:skip:end,1));
				W_EI = squeeze(W_grid(nee,nei,nie,nii,1:skip:end,2));
				W_IE = squeeze(W_grid(nee,nei,nie,nii,1:skip:end,3));
				W_II = squeeze(W_grid(nee,nei,nie,nii,1:skip:end,4));

				if ~isnan(W_EE)
					W_EE(W_EE>W_EE_lims(2)) = NaN;	% prevent plotting outside of limits
					plot3(W_EE,W_IE,W_EI,'g-','linewidth',lwidth);
					hold on;
					plot3(W_EE(1),W_IE(1),W_EI(1),'go','markersize',msize);
					plot3(W_EE(end),W_IE(end),W_EI(end),'g.','markersize',msize);
				else
					plot3(W_EE,W_IE,W_EI,'k-','linewidth',lwidth);
					hold on;
					plot3(W_EE(1),W_IE(1),W_EI(1),'ko');
					plot3(W_EE(end),W_IE(end),W_EI(end),'k.');
				end
			end
		end
	end
end
[x,y] = meshgrid(W_EEinigrid,W_IEinigrid);
W_EIplane = W_EIup(x);
W_IIplane = W_IIup(y);
% W_EIdetcond_3D_x = W_EEinigrid;
% W_EIdetcond_3D_y = W_IEdetcond(W_EEinigrid);
% W_EIdetcond_3D_z = W_EIup(W_EIdetcond_3D_x);
% W_EItrcond_3D_x = W_EEinigrid;
% W_EItrcond_3D_y = W_IEtrcond(W_EEinigrid);
% W_EItrcond_3D_z = W_EIup(W_EItrcond_3D_x);

grid_step = 0.2;
W_EEinigrid2 = [grid_step:grid_step:W_EEinigrid(end)];
W_IEinigrid2 = [grid_step:grid_step:W_IEinigrid(end)];
[x2,y2] = meshgrid(W_EEinigrid2,W_IEinigrid2);
W_EIplane2 = W_EIup(x2);

W_IEdetcond_x2 = W_EEinigrid2;
W_IEdetcond_y2 = W_IEdetcond(W_EEinigrid2);
W_IEtrcond_x2 = W_EEinigrid2;
W_IEtrcond_y2 = W_IEtrcond(W_EEinigrid2);
intersect_det_tr_idx = find(W_IEdetcond_y2 > W_IEtrcond_y2,1);

stable_region_1 = [W_EEinigrid2(intersect_det_tr_idx:end); W_IEdetcond(W_EEinigrid2(intersect_det_tr_idx:end))];
stable_region_2 = [W_EEinigrid2(end); W_IEinigrid2(end)];
stable_region_3 = fliplr([W_EEinigrid2(intersect_det_tr_idx:end); W_IEtrcond(W_EEinigrid2(intersect_det_tr_idx:end))]);
stable_region = [stable_region_1, stable_region_2, stable_region_3]';
[in,on] = inpolygon(x2(:),y2(:),stable_region(:,1),stable_region(:,2));
mask = reshape(in|on,size(x2));
W_EIplane2_stable = W_EIplane2;
W_EIplane2_stable(~mask) = NaN;
W_EIplane2_stable(W_EIplane2_stable<0) = NaN;
W_EIplane2_unstable = W_EIplane2;
W_EIplane2_unstable(mask) = NaN;
W_EIplane2_unstable(W_EIplane2_unstable<0) = NaN;

% pts = surf(x,y,W_EIplane);
pts1 = surf(x2,y2,W_EIplane2_stable);
% set(pts,'facecolor',[0.95 0.75 0.95]);
set(pts1,'facecolor',[0.5 0.75 0.5]);
% set(pts1,'clipping','on','cippingstyle','3dbox');
% set(pts,'edgecolor','none');
pts2 = surf(x2,y2,W_EIplane2_unstable);
set(pts2,'facecolor',[0.95 0.95 0.95]);
% set(pts2,'clipping','on','cippingstyle','3dbox');

% ptl_det = plot3(W_EIdetcond_3D_x,W_EIdetcond_3D_y,W_EIdetcond_3D_z,'g-','linewidth',lwidth);
% ptl_tr = plot3(W_EItrcond_3D_x,W_EItrcond_3D_y,W_EItrcond_3D_z,'g--','linewidth',lwidth);

view([-58 14]);
% view([-220 8]);
xlabel('W_{EE}');
ylabel('W_{IE}');
zlabel('W_{EI}');
xlim(W_EE_lims);
ylim(W_IE_lims);
zlim(W_EI_lims);
box on;



%% Plot full grid 3D - PCA


lwidth = 2;
msize = 10;


figure(5);
clf(5)



skip = 2;
for nee = 1:2:NEE
	for nei = 1:2:NEI
		for nie = 1:2:NIE
			for nii = 1%1:2:NII
				W_EE = squeeze(W_grid_pca(nee,nei,nie,nii,1:skip:end,1));
				W_EI = squeeze(W_grid_pca(nee,nei,nie,nii,1:skip:end,2));
				W_IE = squeeze(W_grid_pca(nee,nei,nie,nii,1:skip:end,3));
				W_II = squeeze(W_grid_pca(nee,nei,nie,nii,1:skip:end,4));

				if ~isnan(W_EE)
					plot3(W_EE,W_IE,W_EI,'m-','linewidth',lwidth);
					hold on;
					plot3(W_EE(1),W_IE(1),W_EI(1),'mo','markersize',msize);
					plot3(W_EE(end),W_IE(end),W_EI(end),'m.','markersize',msize);
				else
					plot3(W_EE,W_IE,W_EI,'k-','linewidth',lwidth);
					hold on;
					plot3(W_EE(1),W_IE(1),W_EI(1),'ko');
					plot3(W_EE(end),W_IE(end),W_EI(end),'k.');
				end
				
			end
		end
	end
end

box on;



%%
[x,y] = meshgrid(W_EEinigrid,W_IEinigrid);
W_EIplane = W_EIup(x);
W_IIplane = W_IIup(y);
pts = surf(x,y,W_EIplane);
set(pts,'facecolor',[0.85 0.85 0.85]);
% set(pts,'edgecolor','none');

xlabel('W_{EE}');
ylabel('W_{IE}');
zlabel('W_{EI}');
xlim(W_EE_lims);
ylim(W_IE_lims);
zlim(W_EI_lims);
box on;




%%
