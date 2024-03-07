%% PQD GENERATION (NORMAL + 19 PQDS) 
% 1000 SIGNALS FOR EACH PQD WITH 640 SAMPLE POINTS (1000 signals/class /SNR)
% TRAIN DATA = 70%, VALIDATION = 20% TEST DATA = 10%

clc
clear all

ts = 1/3200;
z=[]; cl=[]; X_train=[]; Y_train=[]; X_test=[]; Y_test=[]; X_val=[]; Y_val=[];
SNR = [20, 30, 40, 100];
snr_len = length(SNR);
iter_disp = 'Iteration count ';

%% Normal
x = "Normal";
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_normal = {'Sine Wave with 20db Noise','Sine Wave with 30db Noise','Sine Wave with 40db Noise','Pure Sine wave'};

counter=0;
for i = 1:snr_len
    for f = 49.5:0.0250626566:50.5                % 1000 different sine waves  (Runs 40 times)
        for vm=0.95:0.0040160643:1.05         % (Runs 25 times)
            counter=counter+1;
            if(rem(counter,500)==0)
                disp(iter_disp)
                disp(counter)
            end
            y = vm*sin(2*pi*f*t);
            y =awgn(y, SNR(i));
            z= vertcat(z,y);
            cl=vertcat(cl,x);
        end
    end
    figure(i)
    plot(t,y)
    title(fig_normal(i))
end

matrix = z;

% Define the matrix
[m, n] = size(matrix); % assuming 'matrix' is your m by n matrix

% Initialize cell arrays for train, val, and test
sine_train = [];
sine_val = [];
sine_test = [];

% Loop through the matrix in chunks of 10 rows
for i = 1:10:m
    chunk = matrix(i:min(i+9, m), :);  % Get a chunk of 10 rows or less
    
    % Split the chunk into train, val, and test
    train_chunk = chunk(1:min(7, size(chunk, 1)), :);
    val_chunk = chunk(min(7, size(chunk, 1))+1:min(9, size(chunk, 1)), :);
    test_chunk = chunk(end, :);
    
    % Append the chunks to their respective lists
    %list_train = [list_train; {train_chunk}];
    sine_train = vertcat(sine_train, train_chunk);
    %list_val = [list_val; {val_chunk}];
    sine_val = vertcat(sine_val, val_chunk);
    %list_test = [list_test; {test_chunk}];
    sine_test = vertcat(sine_test, test_chunk);
end

% Save the matrices into a .mat file
save('Sine_data.mat', 'sine_train', 'sine_val', 'sine_test');

"""
% Split the list into chunks of 10 elements.
% And split the first 7 elements into train, next 2 elements into val, and the last element into test. 
% Define the original list
list = 1:20;

% Initialize lists for train, val, and test
list_train = [];
list_val = [];
list_test = [];

% Loop through the list in chunks of 10 elements
for i = 1:10:numel(list)
    chunk = list(i:min(i+9, end));  % Get a chunk of 10 elements or less
    
    % Split the chunk into train, val, and test
    train_chunk = chunk(1:min(7, numel(chunk)));
    val_chunk = chunk(min(7, numel(chunk))+1:min(9, numel(chunk)));
    test_chunk = chunk(end);
    
    % Append the chunks to their respective lists
    list_train = [list_train, train_chunk];
    list_val = [list_val, val_chunk];
    list_test = [list_test, test_chunk];
end

% Display the results
disp('Train list:');
disp(list_train);
disp('Validation list:');
disp(list_val);
disp('Test list:');
disp(list_test);
"""




%% Sag
fig_sag = {'Sag disturbance with 20db Noise','Sag disturbance with 30db Noise','Sag disturbance with 40db Noise','Sag disturbance with No Noise'};
x = "Sag";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
del_t = [0.1, 0.09, 0.06];

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.1:0.0376884422:0.85              % Runs 20 times
        for t1=0.04:0.0007228916:0.058              % Runs 25 times
            for f=49.9:0.1:50               % Runs 2 times
            counter=counter+1;
            if(rem(counter,500)==0)
                disp(iter_disp)
                disp(counter)
            end
                count=count+1;              
                [index,remin] = quorem(sym(count),sym(334));
                y=(1- alpha*(heaviside(t-t1)-heaviside(t-(t1+del_t(index+1))))).*sin(2*pi*f*t); 
                y =awgn(y, SNR(i));
                z=vertcat(z,y);
                cl=vertcat(cl,x);
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_sag(i))
end

%% Swell

x = "Swell";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_swell = {'Swell disturbance with 20db Noise','Swell disturbance with 30db Noise','Swell disturbance with 40db Noise','Swell disturbance with No Noise'};
del_t = [0.1, 0.09, 0.06];

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.1:0.0351758794:0.8                   % Runs 20 times
        for t1=0.04:0.0007228916:0.058                       % Runs 25 times
            for f=49.9:0.1:50                        % Runs 2 times
            counter=counter+1;
            if(rem(counter,500)==0)
                disp(iter_disp)
                disp(counter)
            end
              count=count+1;
                [index,remin] = quorem(sym(count),sym(334));
                y=(1+ alpha*(heaviside(t-t1)-heaviside(t-(t1+del_t(index+1))))).*sin(2*pi*f*t);
                y = awgn(y, SNR(i));
                z=vertcat(z,y);
                cl=vertcat(cl,x);
        
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_swell(i))
end

%% Interruption

x = "Interruption";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_interruption = {'Interruption disturbance with 20db Noise','Interruption disturbance with 30db Noise','Interruption disturbance with 40db Noise','Interruption disturbance with No Noise'};
del_t = [0.1, 0.09, 0.06];

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.9:0.0050251256:1                    % Runs 20 times
        for t1=0.04:0.0007228916:0.058                      % Runs 25 times
            for f=49.9:0.1:50                       % Runs 2 times
            counter=counter+1;
            if(rem(counter,500)==0)
                disp(iter_disp)
                disp(counter)
            end            
            count=count+1;
                [index,remin] = quorem(sym(count),sym(334));
                y=(1-alpha*(heaviside(t-t1)-heaviside(t-(t1+del_t(index+1))))).*sin(2*pi*f*t);
                y = awgn(y, SNR(i));
                z= vertcat(z,y);
                cl=vertcat(cl,x); 
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_interruption(i))
end

%% Harmonics

x = "Harmonics";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_harmonics = {'Harmonics disturbance with 20db Noise','Harmonics disturbance with 30db Noise','Harmonics disturbance with 40db Noise','Harmonics disturbance with No Noise'};

counter=0;
for i = 1:snr_len
    h1=7;
    count=0;
    for alpha3=0.05:0.0040160643:0.15            % Runs 25 times  
        for frac1=0.5:0.1:0.8                     % Runs 4 times
           alpha5 = alpha3*frac1;
            for frac2=0.4:0.1:0.8                     % Runs 5 times
                alpha7 = alpha5*frac2;
                for f=49.95:0.1:50.1                         % Runs 2  times
                    counter=counter+1;
                    if(rem(counter,500)==0)
                        disp(iter_disp)
                        disp(counter)
                    end
                    count=count+1;
                    if(rem(count,125)==0)
                        h1=h1+2;   %%7,9,11,13,15,17,19,21
                    end
                alpha1 = sqrt(1 - alpha3^2 - alpha5^2- alpha7^2);
                y = alpha1*sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(10*pi*f*t)+ alpha7*sin(2*h1*pi*f*t);
                y = awgn(y, SNR(i));
                z= vertcat(z,y);
                cl=vertcat(cl,x); 
            end
        end
    end
    end
    figure(i)
    plot(t,y)
    title(fig_harmonics(i))
end

%% Flicker

fig_flicker = {'Flicker disturbance with 20db Noise','Flicker disturbance with 30db Noise','Flicker disturbance with 40db Noise','Flicker disturbance with No Noise'};
x = "Flicker";                            
t = [0: ts :0.2-ts];                            % 640 sample points per disturbance

counter=0;
for i = 1:snr_len
    for alpha_flicker=0.06:0.007035175879:0.2       % Runs 20 times
        for beta=8:0.6827309237:25                  % Runs 25 times
            for f=49.9:0.1:50                         % Runs 2 times
            counter=counter+1;
            if(rem(counter,500)==0)
                disp(iter_disp)
                disp(counter)
            end
                y=(1+alpha_flicker*sin(beta*2*pi*f*t)).*sin(2*pi*f*t);
                y = awgn(y, SNR(i));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_flicker(i))
end
        

%% Oscillatory Transient

x = "Oscillatory Transient";                            
t = [0: ts :0.2-ts];                                        % 640 sample points per disturbance
fig_Oscillatory_Transient = {'Oscillatory Transient disturbance with 20db Noise','Oscillatory Transient disturbance with 30db Noise','Oscillatory Transient disturbance with 40db Noise','Oscillatory Transient disturbance with No Noise'};
t4i=[0.02, 0.03, 0.04];

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.1:0.1428571429:0.8                          % Runs 5 times
        for F_t=250:300:1500                     % Runs 5 times
            for t3=0.015:0.0138888:0.14                 % Runs 10 times
                for tau=0.008:0.025:0.040            % Runs 2 times
                    for f=49.9:0.1:50                       % Runs 2 times
                    counter=counter+1;
                    if(rem(counter,500)==0)
                         disp(iter_disp)
                        disp(counter)
                    end
                        count=count+1;
                        [index,remin] = quorem(sym(count),sym(334));
                        t4=t3+t4i(index+1);          % 1 cycle
                        y= sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                     
                    end
                end
            end 
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Oscillatory_Transient(i))
end

%% Notch

x = "Notch";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Notch = {'Notch disturbance with 20db Noise','Notch disturbance with 30db Noise','Notch disturbance with 40db Noise','Notch disturbance with No Noise'};

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.15:0.0100401606:0.4            % Runs 25 times
        for t1=0.0005:0.000944:0.009        % Runs 10 times
              for f=49.95:0.05:50.1                        % Runs 4 times
              counter=counter+1;
                if(rem(counter,500)==0)
                    disp(iter_disp)
                    disp(counter)
                end
              count=count+1;
                if (count<500)
                    t2=t1+0.0005;      % 0.025 cycle
                    sum = 0;
                    for n=0:9
                        sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
                    end
                    y = sin(2*pi*f*t) - alpha*sign(2*pi*f*t).*sum;
                    y = awgn(y, SNR(i));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
                else       
                    t2=t1+0.001;       % 0.05 cycle
                    sum = 0;
                    for n=0:9
                        sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
                    end
                    y = sin(2*pi*f*t) - alpha*sign(2*pi*f*t).*sum;
                    y = awgn(y, SNR(i));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
                end
             end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Notch(i))
end

%% Spike

x = "Spike";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Spike = {'Spike disturbance with 20db Noise','Spike disturbance with 30db Noise','Spike disturbance with 40db Noise','Spike disturbance with No Noise'};

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.15:0.0100401606:0.4;            % Runs 25 times
        for t1=0.0005:0.000944:0.009;        % Runs 10 times
              for f=49.95:0.05:50.1                        % Runs 4 times
              counter=counter+1;
                if(rem(counter,500)==0)
                    disp(iter_disp)
                    disp(counter)
                end
              count=count+1;
                if (count<500)
                    t2=t1+0.0005;      % 0.025 cycle
                    sum = 0;
                    for n=0:9
                        sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
                    end
                    y = sin(2*pi*f*t) + alpha*sign(2*pi*f*t).*sum;
                    y = awgn(y, SNR(i));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
                else       
                    t2=t1+0.001;       % 0.05 cycle
                    sum = 0;
                    for n=0:9
                        sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
                    end
                    y = sin(2*pi*f*t) + alpha*sign(2*pi*f*t).*sum;
                    y = awgn(y, SNR(i));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
                end
             end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Spike(i))
end



%% Sag + Harmonics

x = "Sag+Harmonics";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Sag_Harmonics = {'Sag+Harmonics disturbance with 20db Noise','Sag+Harmonics disturbance with 30db Noise','Sag+Harmonics disturbance with 40db Noise','Sag+Harmonics disturbance with No Noise'};
t2i=[0.1, 0.09, 0.06];

counter=0;
for i = 1:snr_len
    count=0;
    h1=7;
    for alpha=0.1:0.26:0.9                         % Runs 4 times  
        for t1=0.04:0.0036734694:0.058                    % Runs 5 times
            for f=49.9:0.1:50                        % Runs 2 times 
                for alpha3=0.05:0.025:0.15               % Runs 5 times
                    for frac5=0.4:0.1:0.8       % Runs 5 times
                        alpha5=frac5*alpha3;
                        alpha7=frac5*alpha5;
                        alpha1 = sqrt(1 - alpha3^2 - alpha5^2- alpha7^2);
                        counter=counter+1;
                        if(rem(counter,500)==0)
                            disp(iter_disp)
                            disp(counter)
                        end
                        count=count+1;
                        [index,remin] = quorem(sym(count),sym(334));
                        
                        if(rem(count,125)==0)
                            h1=h1+2;   %%7,9,11,13,15,19,21
                        end
                        
                        t2=t1+t2i(index+1);                %5 cycles
                        y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1*sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t)+ alpha7*sin(h1*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                       
                    end
                end
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Sag_Harmonics(i))
end

%% Swell + Harmonics

x = "Swell+Harmonics";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Swell_Harmonics = {'Swell+Harmonics disturbance with 20db Noise','Swell+Harmonics disturbance with 30db Noise','Swell+Harmonics disturbance with 40db Noise','Swell+Harmonics disturbance with No Noise'};
t2i=[0.1, 0.09, 0.06];

counter=0;
for i = 1:snr_len
    count=0;
    h1=3;
    h2=5;
    for alpha=0.1:0.1794871795:0.8                         % Runs 4 times  
        for t1=0.04:0.0036734694:0.058                    % Runs 5 times
            for f=49.9:0.1:50                        % Runs 2 times
                for alpha3=0.05:0.025:0.15               % Runs 5 times
                    for alpha5=0.05:0.0204081633:0.15       % Runs 5 times
                        alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
                        counter=counter+1;
                        if(rem(counter,500)==0)
                            disp(iter_disp)
                            disp(counter)
                        end
                        count=count+1;
                        [index,remin] = quorem(sym(count),sym(334));
                        
                        if(rem(count,200)==0)
                            h1=h1+4;   %%3,7,11,15,19
                            h2=h2+4;     %5,9,13,17,21
                        end
                        
                        t2=t1+t2i(index+1);                %5,4.5,3 cycles
                        y = (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(h1*2*pi*f*t)+ alpha5*sin(h2*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                    end
                end
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Swell_Harmonics(i))
end

%% Interruption + Harmonics
%% same formula as sag+harmonic

x = "Interruption+Harmonics";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Interruption_Harmonics = {'Interruption+Harmonics disturbance with 20db Noise','Interruption+Harmonics disturbance with 30db Noise','Interruption+Harmonics disturbance with 40db Noise','Interruption+Harmonics disturbance with No Noise'};
t2i=[0.1, 0.09, 0.06];

counter=0;
for i = i:snr_len
    count=0;
    h1=3;
    h2=5;
    for alpha=0.9:0.0256410256:1                           % Runs 4 times  
        for t1=0.04:0.0036734694:0.058                    % Runs 5 times
            for f=49.9:0.1:50                        % Runs 2 times
                for alpha3=0.05:0.025:0.15               % Runs 5 times
                    for alpha5=0.05:0.0204081633:0.15       % Runs 5 times
                        alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
                        counter=counter+1;
                        if(rem(counter,500)==0)
                            disp(iter_disp)
                            disp(counter)
                        end
                        count=count+1;
                        [index,remin] = quorem(sym(count),sym(334));

                        if(rem(count,200)==0)
                            h1=h1+4;   %%3,7,11,15,19
                            h2=h2+4;     %5,9,13,17,21
                        end

                        t2=t1+t2i(index+1);                %5,4.5,3 cycles
                        y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(h1*2*pi*f*t)+ alpha5*sin(h2*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                    end
                end
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Interruption_Harmonics(i))
end

%% Flicker + Harmonics

x = "Flicker+Harmonics";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Flicker_Harmonics = {'Flicker+Harmonics disturbance with 20db Noise','Flicker+Harmonics disturbance with 30db Noise','Flicker+Harmonics disturbance with 40db Noise','Flicker+Harmonics disturbance with No Noise'};

counter=0;
for i = 1:snr_len
    count=0;
    for alpha_flicker=0.08:0.0307692308:0.2                 % Runs 4 times
        for beta=5:3.703703704:20                          % Runs 5 times
            for f=49.9:0.1:50                        % Runs 2 times
                for alpha3=0.05:0.025:0.15               % Runs 5 times
                    for alpha5=0.05:0.0204081633:0.15           % Runs 5 times 
                        alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
                        counter=counter+1;
                        if(rem(counter,500)==0)
                            disp(iter_disp)
                            disp(counter)
                        end
                        count=count+1;
                        if(rem(count,200)==0)
                            h1=h1+4;   %%3,7,11,15,19
                            h2=h2+4;     %5,9,13,17,21
                        end
                        y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(h1*2*pi*f*t)+ alpha5*sin(h2*2*pi*f*t));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x); 
                    end
                end
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Flicker_Harmonics(i))
end

%% Flicker + Sag

x = "Flicker+Sag";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Flicker_Sag = {'Flicker+Sag disturbance with 20db Noise','Flicker+Sag disturbance with 30db Noise','Flicker+Sag disturbance with 40db Noise','Flicker+Sag disturbance with No Noise'};
t2i=[0.1, 0.09, 0.06];

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.1:0.2051282051:0.9                                % Runs 4 times  
        for t1=0.04:0.0036734694:0.058                            % Runs 5 times
            for f=49.9:0.1:50                                     % Runs 2 times
                for alpha_flicker=0.08:0.0244897959:0.2           % Runs 5 times
                    for beta=5:3.703703704:20                     % Runs 5 times 
                        counter=counter+1;
                        if(rem(counter,500)==0)
                            disp(iter_disp)
                            disp(counter)
                        end
                        count=count+1;
                        [index,remin] = quorem(sym(count),sym(334));
                        
                        t2=t1+t2i(index+1);                %5,4.5,3 cycles
                        y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*sin(2*pi*f*t).*(1-alpha*((heaviside(t-t1)-heaviside(t-t2))));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                        
                    end
                end
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Flicker_Sag(i))
end

%% Flicker + Swell

x = "Flicker+Swell";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Flicker_Swell = {'Flicker+Swell disturbance with 20db Noise','Flicker+Swell disturbance with 30db Noise','Flicker+Swell disturbance with 40db Noise','Flicker+Swell disturbance with No Noise'};
t2i=[0.1, 0.09, 0.06];

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.1:0.1794871795:0.8                          % Runs 4 times  
        for t1=0.04:0.0036734694:0.058                      % Runs 5 times
            for f=49.9:0.1:50                        % Runs 2 times
                for alpha_flicker=0.08:0.0244897959:0.2          % Runs 5 times
                    for beta=5:3.703703704:20                      % Runs 5 times 
                        counter=counter+1;
                        if(rem(counter,500)==0)
                            disp(iter_disp)
                            disp(counter)
                        end
                        count=count+1;
                        [index,remin] = quorem(sym(count),sym(334));
                        
                        t2=t1+t2i(index+1);                %5,4.5,3 cycles
                        y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*sin(2*pi*f*t).*(1+alpha*((heaviside(t-t1)-heaviside(t-t2))));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                     
                    end
                end
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Flicker_Swell(i))
end

%% Osciallatory + Sag

x = "Oscillatory Transient + Sag";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Osciallatory_Sag = {'Osciallatory+Sag disturbance with 20db Noise','Osciallatory+Sag disturbance with 30db Noise','Osciallatory+Sag disturbance with 40db Noise','Osciallatory+Sag disturbance with No Noise'};
t4i=[0.02, 0.03, 0.04];

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.1:0.2051282051:0.9                         % Runs 4 times
        for F_t=300:1160.49382:5000                    % Runs 5 times
            for f=49.9:0.1:50                        % Runs 2 times
                for t3=0.04:0.01:0.08                   % Runs 5 times
                    for tau=0.008:0.007901234568:0.040          % Runs 5 times
                        counter=counter+1;
                        if(rem(counter,500)==0)
                            disp(iter_disp)
                            disp(counter)
                        end
                        count=count+1;
                        [index,remin] = quorem(sym(count),sym(334));
                        t4=t3+t4i(index+1);          % 1,1.5,2 cycle
                        y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) * (alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                    end
                end
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Osciallatory_Sag(i))
end

%% Osciallatory + Swell

x = "Oscillatory Transient + Swell ";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Osciallatory_Swell = {'Osciallatory+Swell disturbance with 20db Noise','Osciallatory+Swell disturbance with 30db Noise','Osciallatory+Swell disturbance with 40db Noise','Osciallatory+Swell disturbance with No Noise'};
t4i=[0.02, 0.03, 0.04];

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.1:0.2051282051:0.9                         % Runs 4 times
        for F_t=300:1160.493827:5000                    % Runs 5 times
            for f=49.9:0.1:50                        % Runs 2 times
                for t3=0.04:0.01:0.08                   % Runs 5 times
                    for tau=0.008:0.007901234568:0.040          % Runs 5 times
                        counter=counter+1;
                        if(rem(counter,500)==0)
                            disp(iter_disp)
                            disp(counter)
                        end
                        count=count+1;
                        [index,remin] = quorem(sym(count),sym(334));

                        t4=t3+t4i(index+1);          % 1,1.5,2 cycle
                        y= (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))) * (alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                    end
                end
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Osciallatory_Sag(i))
end

%% Oscillatory Transients + Interruptions

x = "Oscillatory Transient + Interruptions";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_Osciallatory_Interruptions = {'Osciallatory+Interruptions disturbance with 20db Noise','Osciallatory+Interruptions disturbance with 30db Noise','Osciallatory+Interruptions disturbance with 40db Noise','Osciallatory+Interruptions disturbance with No Noise'};
t4i=[0.02, 0.03, 0.04];

counter=0;
for i = 1:snr_len
    count=0;
    for alpha=0.1:0.2051282051:0.9                         % Runs 4 times
        for F_t=300:1160.493827:5000                    % Runs 5 times
            for f=49.9:0.1:50                        % Runs 2 times
                for t3=0.04:0.01:0.08                   % Runs 5 times
                    for tau=0.008:0.007901234568:0.040          % Runs 5 times
                        counter=counter+1;
                        if(rem(counter,500)==0)
                            disp(iter_disp)
                            disp(counter)
                        end
                        count=count+1;
                        [index,remin] = quorem(sym(count),sym(334));

                        t4=t3+t4i(index+1);          % 1,1.5,2 cycle
                        y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) * (alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                    end
                end
            end 
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Osciallatory_Interruptions(i))
end

%% Osciallatory + Harmonics 
% Ask Sir

x = "Oscillatory Transient + Harmonics";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;
fig_Osciallatory_Harmonics = {'Osciallatory+Harmonics disturbance with 20db Noise','Osciallatory+Harmonics disturbance with 30db Noise','Osciallatory+Harmonics disturbance with 40db Noise','Osciallatory+Harmonics disturbance with No Noise'};


for i = 1:snr_len
    for alpha=0.1:0.0773480663:0.9                         % Runs 10 times
        for F_t=300:1160.493827:5000                    % Runs 5 times
            for t3=0.04:0.01:0.08                   % Runs 5 times
                for tau=0.008:0.007901234568:0.040          % Runs 5 times
                    for alpha3=0.05:0.002038735:0.15           % Runs 50 times
                        for alpha5=0.05:0.004158004158:0.15       % Runs 25 times
                            t4=t3+0.02;          % 1 cycle
                            y= (sin(2*pi*f*t)+ alpha*((heaviside(t-t1)-heaviside(t-t2))).exp(t1-t/tau).*sin(2*pi*F_t*t)) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4));
                            y = awgn(y, SNR(i));
                            z= vertcat(z,y);
                            cl=vertcat(cl,x);
                            %{               
                            t4=t3+0.03;          % 1.5 cycles
                            y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                            y = awgn(y, SNR(i));
                            z= vertcat(z,y);
                            cl=vertcat(cl,x);

                            t4=t3+0.04;          %2 cycles
                            y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                            y = awgn(y, SNR(i));
                            z= vertcat(z,y);
                            cl=vertcat(cl,x);
                            %}
                       end
                   end
               end
           end 
       end
    end
    figure(i)
    plot(t,y)
    title(fig_Osciallatory_Harmonics(i))
end
   
%% Inter-Harmonics

x = "Inter-Harmonics";                            
t = [0:t_s:0.2];
f=50;
fig_Inter_Harmonics = {'Inter-Harmonics disturbance with 20db Noise','Inter-Harmonics disturbance with 30db Noise','Inter-Harmonics disturbance with 40db Noise','Inter-Harmonics disturbance with No Noise'};

%%ask sir about beta values
B1= 2;
B2= 5;
for i = 1:snr_len
    for alpha1=0.05:0.0100200401:0.15           % Runs 10 times
        for alpha2=0.05:0.0100200401:0.15       % Runs 10 times
            for alpha3=0.05:0.0200803213:0.15       % Runs 5 times
            y = alpha1*sin(2*pi*f*t)+ alpha2*sin(B1*2*pi*f*t)+ alpha3*sin(B2*2*pi*f*t);
            y = awgn(y, SNR(i));
            z= vertcat(z,y);
            cl=vertcat(cl,x);  
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Inter_Harmonics(i))
end
