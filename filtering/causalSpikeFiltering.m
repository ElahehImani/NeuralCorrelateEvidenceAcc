function [y]=causalSpikeFiltering(spikes,filterOpt,times)
dt=times(2)-times(1);
y=NaN(1,length(times));
newWidth=round(filterOpt.sigma/dt);%convert ms to number of samples

switch(filterOpt.kernel)
    case 'boxcar'
        filt=1/newWidth*ones(1,newWidth);
        y = filter(filt,1,spikes);
        
    case 'half-gaussian'
        mu = 0;
        pd_normal = makedist('Normal',mu,newWidth);
        pd_half_normal = truncate(pd_normal,0,inf);
        X = (-newWidth*4:1:newWidth*4)';
        Xt = X(X>0);
        filt = pdf(pd_half_normal,Xt);
        y=filter(filt,1,spikes);
        
    case 'exp'
        FS=round(1/dt);
        y = expsmooth(spikes',FS,newWidth);
        y=y'; 
end
end