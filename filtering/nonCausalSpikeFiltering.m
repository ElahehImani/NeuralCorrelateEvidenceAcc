function [y]=nonCausalSpikeFiltering(spikes,kernelOpt,times)
mu=(times(end)+times(1))/2;

switch(kernelOpt.kernel)
    case 'boxcar'
        f=abs(times-mu)<kernelOpt.sigma/2;
        f = f ./ kernelOpt.sigma;
        kf = fft(f);
        y = ifftshift(ifft(fft(spikes).*kf));
        
%     case 'gaussian'
%         s2=kernelOpt.sigma^2;
%         f = exp(-0.5*((times - mu).^2)./s2);
%         f = f ./ sqrt(2*pi*s2);
%         kf = fft(f);
%         y = ifftshift(ifft(fft(spikes).*kf));
        
    case 'gaussian'
        y = gaussfilt(times,spikes,kernelOpt.sigma);
end
end