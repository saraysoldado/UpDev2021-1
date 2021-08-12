%% Two-Factor learning rule

function dWdt = kernel_gradDescent(t,W,f_up,params)

	E_set = params.E_set;
	I_set = params.I_set;
	alpha = params.alpha;
	beta_E = params.beta_E;
	beta_I = params.beta_I;
	E = f_up{1}(W(1),W(2),W(3),W(4));
	I = f_up{2}(W(1),W(2),W(3),W(4));

	if (E>0 && I>0)
		dWEEdt = alpha*E*(I_set - I) + beta_E*E*(E_set - E);
		if (W(1)<=0 && dWEEdt<=0)	% if WEE<0 stop updating, unless dWEEdt>0
			dWEEdt = 0;
		end
		dWEIdt = -alpha*I*(I_set - I) - beta_E*I*(E_set - E);
		if (W(2)<=0 && dWEIdt<=0)	% if WEI<0 stop updating, unless dWEIdt>0
			dWEIdt = 0;
		end
		dWIEdt = -alpha*E*(E_set - E) - beta_I*E*(I_set - I);
		if (W(3)<=0 && dWIEdt<=0)	% if WIE<0 stop updating, unless dWIEdt>0
			dWIEdt = 0;
		end
		dWIIdt = alpha*I*(E_set - E) + beta_I*I*(I_set - I);
		if (W(4)<=0 && dWIIdt<=0)	% if WII<0 stop updating, unless dWIIdt>0
			dWIIdt = 0;
		end
		dWdt = [dWEEdt,dWEIdt,dWIEdt,dWIIdt];
	else
		dWdt = [NaN,NaN,NaN,NaN];
	end
end


%%
