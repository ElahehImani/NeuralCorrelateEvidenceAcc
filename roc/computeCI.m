function [CI]=computeCI(data,method,alpha,oneSided)
switch(method)
    case 'percentile'
        ds=sort(data);
        lb=alpha/2;
        ub=1-lb;
        thresh1=fix(lb*length(ds));
        thresh2=ceil(ub*length(ds));
        CI=[ds(thresh1),ds(thresh2)];
end
end