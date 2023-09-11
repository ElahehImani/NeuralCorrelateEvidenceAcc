function [spikeRate]=SpikeRate(spikeTimes,area,boundary,dt)
unique_area=unique(area);
spikeRate={};
count=1;
smpRate=1/dt;
for i=1 : size(spikeTimes,1)
    if(iscell(boundary))
        if(boundary{i}(1)<boundary{i}(2))
            times=boundary{i}(1):dt:boundary{i}(2);
            timeBins=boundary{i}(1)+dt/2 : dt : boundary{i}(2)-dt/2;
        else
            times=boundary{i}(1):-dt:boundary{i}(2);
            timeBins=boundary{i}(1)+dt/2 : -dt : boundary{i}(2)-dt/2;
        end
    else
        if(boundary(1)<boundary(2))
            times=boundary(1):dt:boundary(2);
            timeBins=boundary(1)+dt/2 : dt : boundary(2)-dt/2;
        else
            times=boundary(1):-dt:boundary(2);
            timeBins=boundary(1)+dt/2 : -dt : boundary(2)-dt/2;
        end
    end

    c=1;
    for j=1 : length(unique_area)
        if(strcmp(unique_area{j},'invalid'))
            continue;
        end
        cluster_idx=strcmp(area,unique_area{j});

        st_cell=spikeTimes(i,cluster_idx);
        rate = zeros(length(timeBins),length(st_cell));
        for k=1 : length(st_cell)
            st=st_cell{k};
            if(~isempty(st))
                rate((floor((st - times(1))*smpRate) + 1),k) = 1;
            end
        end

        rate=rate';
        if(size(rate,2)==1)
            rate=rate';
        end
        spikeRate{i,c}=rate;

        c=c+1;
    end

    count=count+1;
end
end