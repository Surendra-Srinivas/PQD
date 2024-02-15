%% PQD GENERATION (NORMAL + 19 PQDS) 
% 1000 SIGNALS FOR EACH PQD WITH 640 SAMPLE POINTS (1000 signals/class /SNR)
% TRAIN DATA = 70%, VALIDATION = 20% TEST DATA = 10%

clc
clear all

ts = 1/3200;
t1_s = 0.54/25;
z=[]; cl=[]; X_train=[]; Y_train=[]; X_test=[]; Y_test=[]; X_val=[]; Y_val=[];
SNR = [10, 20, 30, 40, 100];
snr_len = length(SNR);
fig_normal = {'Sine Wave with 10db Noise','Sine Wave with 20db Noise','Sine Wave with 30db Noise','Sine Wave with 40db Noise','Pure Sine wave'};

%% Normal
for i = 1:snr_len
    x = "Normal";
    t = [0:ts:0.2-ts];                   % 640 sample points per disturbance (1000 different sine waves)  
    for f = 49.5:0.025:50.5                % (Runs 40 times)
        for vm=0.95:0.00416:1.05         % (Runs 25 times)
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


%% Sag
fig_sag = {'Sag disturbance with 10db Noise','Sag disturbance with 20db Noise','Sag disturbance with 30db Noise','Sag disturbance with 40db Noise','Sag disturbance with No Noise'};
x = "Sag";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;
for i = 1:snr_len
    %del_t = [0.1, 0.09, 0.06];
    for alpha=0.1:0.01875:0.85              % Runs 40 times
        for t1=0.04:t1_s:0.058              % Runs 25 times
            y=(1- alpha*((heaviside(t-t1)-heaviside(t-(t1+0.1))))).*sin(2*pi*f*t);  %5 cycles
            y =awgn(y, SNR(i));
            z=vertcat(z,y);
            cl=vertcat(cl,x);
    %{        
            y=(1- alpha*((heaviside(t-t1)-heaviside(t-(t1+0.09))))).*sin(2*pi*f*t); %4.5 cycles
            z=vertcat(z,y);
            cl=vertcat(cl,x);
            
            y=(1- alpha*((heaviside(t-t1)-heaviside(t-(t1+0.06))))).*sin(2*pi*f*t); %3 cycles
            z=vertcat(z,y);
            cl=vertcat(cl,x);
    %}
        end
    end
    figure(i)
    plot(t,y)
    title(fig_sag(i))
end

%% Swell

x = "Swell";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;
fig_swell = {'Swell disturbance with 10db Noise','Swell disturbance with 20db Noise','Swell disturbance with 30db Noise','Swell disturbance with 40db Noise','Swell disturbance with No Noise'};

for i = 1:snr_len
    for alpha=0.1:0.0175:0.8            % Runs 40 times
        for t1=0.04:t1_s:0.058          % Runs 25 times
            y=(1+ alpha*((heaviside(t-t1)-heaviside(t-(t1+0.1))))).*sin(2*pi*f*t);
            y = awgn(y, SNR(i));
            z=vertcat(z,y);
            cl=vertcat(cl,x);
     %{       
            y=(1+ alpha*((heaviside(t-t1)-heaviside(t-(t1+0.09))))).*sin(2*pi*f*t);
            z=vertcat(z,y);
            cl=vertcat(cl,x);
            
            y=(1+ alpha*((heaviside(t-t1)-heaviside(t-(t1+0.06))))).*sin(2*pi*f*t);
            z=vertcat(z,y);
            cl=vertcat(cl,x);
    %}
        end
    end
    figure(i)
    plot(t,y)
    title(fig_swell(i))
end

%% Interruption

x = "Interruption";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;
fig_interruption = {'Interruption disturbance with 10db Noise','Interruption disturbance with 20db Noise','Interruption disturbance with 30db Noise','Interruption disturbance with 40db Noise','Interruption disturbance with No Noise'};
for i = 1:snr_len
    for alpha=0.9:0.00204:1                 % Runs 50 times
        for t1=0.04:t1_s:0.058           % Runs 25 times
            y=(1-alpha*((heaviside(t-t1)-heaviside(t-(t1+0.1))))).*sin(2*pi*f*t);
            y = awgn(y, SNR(i));
            z= vertcat(z,y);
            cl=vertcat(cl,x);
    %{        
            y=(1-alpha*((heaviside(t-t1)-heaviside(t-(t1+0.09))))).*sin(2*pi*f*t);
            z= vertcat(z,y);
            cl=vertcat(cl,x);
                    
            y=(1-alpha*((heaviside(t-t1)-heaviside(t-(t1+0.06))))).*sin(2*pi*f*t);
            z= vertcat(z,y);
            cl=vertcat(cl,x);    
            %}    
        end
    end
    figure(i)
    plot(t,y)
    title(fig_interruption(i))
end

%% Harmonics

x = "Harmonics";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
fig_harmonics = {'Harmonics disturbance with 10db Noise','Harmonics disturbance with 20db Noise','Harmonics disturbance with 30db Noise','Harmonics disturbance with 40db Noise','Harmonics disturbance with No Noise'};

for i = 1:snr_len
    for alpha3=0.05:0.004:0.15                         % Runs 25 times
        for alpha5=0.05:0.005:0.15                     % Runs 20 times
            for f=49.9:0.1:50                          % Runs 2 times
                alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
                y = alpha1*sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*pi*f*t);
                y = awgn(y, SNR(i));
                z= vertcat(z,y);
                cl=vertcat(cl,x); 
            end
        end
    end
    figure(i)
    plot(t,y)
    title(fig_harmonics(i))
end

%% Flicker

fig_flicker = {'Flicker disturbance with 10db Noise','Flicker disturbance with 20db Noise','Flicker disturbance with 30db Noise','Flicker disturbance with 40db Noise','Flicker disturbance with No Noise'};
x = "Flicker";                            
t = [0: ts :0.2-ts];                            % 640 sample points per disturbance
for i = 1:snr_len
    for alpha_flicker=0.06:0.0.007035175879:0.2       % Runs 20 times
        for beta=8:0.0.6827309237:25                  % Runs 25 times
            for f=49.9:0.1:50                         % Runs 2 times
                y=(1+alpha_flicker*sin(beta*2*pi*f*t));
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

fig_Oscillatory_Transient = {'Oscillatory Transient disturbance with 10db Noise','Oscillatory Transient disturbance with 20db Noise','Oscillatory Transient disturbance with 30db Noise','Oscillatory Transient disturbance with 40db Noise','Oscillatory Transient disturbance with No Noise'};
x = "Oscillatory Transient";                            
t = [0: ts :0.2-ts];                                        % 640 sample points per disturbance

for i = 1:snr_len
    for alpha=0.1:0.1428571429:0.8                          % Runs 5 times
        for F_t=300:959.1836734694:5000                     % Runs 5 times
            for t3=0.04:0.008163265306:0.08                 % Runs 5 times
                for tau=0.008:0.01684210526:0.040           % Runs 2 times
                    for f=49.9:0.1:50                       % Runs 2 times
                        t4=t3+0.02;          % 1 cycle
                        y= sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                        %{               
                        t4=t3+0.03;          % 1.5 cycles
                        y= sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                        
                        t4=t3+0.04;          %2 cycles
                        y= sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                        %}
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
f = 50;
fig_Notch = {'Notch disturbance with 10db Noise','Notch disturbance with 20db Noise','Notch disturbance with 30db Noise','Notch disturbance with 40db Noise','Notch disturbance with No Noise'};

for i = 1:snr_len
    for alpha=0.1:0.0006116207951:0.4;            % Runs 50 times
        for t1=0.001:0.00037422037642:0.01;        % Runs 25 times
            t2=t1+0.0005;      % 0.025 cycle
            sum = 0;
            for n=0:9
                sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
            end
            y = sin(2*pi*f*t) - alpha*sign(2*pi*f*t).*sum;
            y = awgn(y, SNR(i));
            z= vertcat(z,y);
            cl=vertcat(cl,x);
    %{        
            t2=t1+0.001;       % 0.05 cycle
            sum = 0;
            for n=0:9
                sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
            end
            y = sin(2*pi*f*t) - alpha*sign(2*pi*f*t).*sum;
            y = awgn(y, SNR(i));
            z= vertcat(z,y);
            cl=vertcat(cl,x);
        %}
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Notch(i))
end

%% Spike

x = "Spike";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;
fig_Spike = {'Spike disturbance with 10db Noise','Spike disturbance with 20db Noise','Spike disturbance with 30db Noise','Spike disturbance with 40db Noise','Spike disturbance with No Noise'};

for i = 1:snr_len
    for alpha=0.1:0.0006116207951:0.4;            % Runs 50 times
        for t1=0.001:0.00037422037642:0.01;        % Runs 25 times
            t2=t1+0.0005;      % 0.025 cycle
            sum = 0;
            for n=0:9
                sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
            end
            y = sin(2*pi*f*t) + alpha*sign(2*pi*f*t).*sum;
            y = awgn(y, SNR(i));
            z= vertcat(z,y);
            cl=vertcat(cl,x);
            %{       
            t2=t1+0.001;       % 0.05 cycle
            sum = 0;
            for n=0:9
                sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
            end
            y = sin(2*pi*f*t) + alpha*sign(2*pi*f*t).*sum;
            z= vertcat(z,y);
            cl=vertcat(cl,x);
            %}
        end
    end
    figure(i)
    plot(t,y)
    title(fig_Spike(i))
end


%% Sag + Harmonics

x = "Sag+Harmonics";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;
fig_Sag_Harmonics = {'Sag+Harmonics disturbance with 10db Noise','Sag+Harmonics disturbance with 20db Noise','Sag+Harmonics disturbance with 30db Noise','Sag+Harmonics disturbance with 40db Noise','Sag+Harmonics disturbance with No Noise'};

for i = 1:snr_len
    for alpha=0.1:0.0888:0.9                         % Runs 10 times  
        for t1=0.04:0.0044444:0.058                    % Runs 5 times
            for alpha3=0.05:0.025:0.15               % Runs 5 times
                    for alpha5=0.05:0.02469135802:0.15       % Runs 5 times
                        alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
                
                        t2=t1+0.1;                %5 cycles
                        y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
    %{            
                        t2=t1+0.09;               %4.5 cycles
                        y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                
                        t2=t1+0.06;               %3 cycles
                        y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                        %}
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
f = 50;
fig_Swell_Harmonics = {'Swell+Harmonics disturbance with 10db Noise','Swell+Harmonics disturbance with 20db Noise','Swell+Harmonics disturbance with 30db Noise','Swell+Harmonics disturbance with 40db Noise','Swell+Harmonics disturbance with No Noise'};

for i = 1:snr_len
    for alpha=0.1:0.0777:0.8                         % Runs 10 times  
        for t1=0.04:0.0044444:0.058                    % Runs 5 times
            for alpha3=0.05:0.025:0.15               % Runs 5 times
                    for alpha5=0.05:0.02469135802:0.15       % Runs 5 times
                        alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
                
                        t2=t1+0.1;                %5 cycles
                        y = (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
    %{            
                        t2=t1+0.09;               %4.5 cycles
                        y = (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                
                        t2=t1+0.06;               %3 cycles
                        y = (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                        %}
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
f = 50;
fig_Interruption_Harmonics = {'Interruption+Harmonics disturbance with 10db Noise','Interruption+Harmonics disturbance with 20db Noise','Interruption+Harmonics disturbance with 30db Noise','Interruption+Harmonics disturbance with 40db Noise','Interruption+Harmonics disturbance with No Noise'};

for i = i:snr_len
    for alpha=0.9:0.0111:1                           % Runs 10 times  
        for t1=0.04:0.0044444:0.058                    % Runs 5 times
            for alpha3=0.05:0.025:0.15               % Runs 5 times
                    for alpha5=0.05:0.02469135802:0.15       % Runs 5 times
                        alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
                
                        t2=t1+0.1;                %5 cycles
                        y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                        %{            
                        t2=t1+0.09;               %4.5 cycles
                        y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                
                        t2=t1+0.06;               %3 cycles
                        y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                        y = awgn(y, SNR(i));
                        z= vertcat(z,y);
                        cl=vertcat(cl,x);
                        %}
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
f = 50;

for alpha_flicker=0.08:0.01104972376:0.2                 % Runs 10 times
    for beta=5:3.703703704:20                          % Runs 5 times
        for alpha3=0.05:0.025:0.15               % Runs 5 times
            for alpha5=0.05:0.02469135802:0.15           % Runs 5 times 
                alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
                y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                z= vertcat(z,y);
                cl=vertcat(cl,x); 
            end
        end
    end
end              
        
figure(13)
plot(t,y)
title('Flicker+Harmonics');

for i=L:length(z)
    if rem(i,10)==0 
        X_test = vertcat(X_test,z(i,:));
        Y_test = vertcat(Y_test,cl(i));
    else
        X_train = vertcat(X_train,z(i,:));
        Y_train = vertcat(Y_train,cl(i));
    end
end
L=length(z)+1;

%% Flicker + Sag

x = "Flicker+Sag";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;

for alpha=0.1:0.088:0.9                          % Runs 10 times  
    for t1=0.04:0.0044444:0.058                      % Runs 5 times
        for alpha_flicker=0.08:0.002469135802:0.2          % Runs 5 times
            for beta=5:3.703703704:20                      % Runs 5 times          
                t2=t1+0.1;                %5 cycles
                y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*sin(2*pi*f*t).*(1-alpha*((heaviside(t-t1)-heaviside(t-t2))));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
%{            
                t2=t1+0.09;               %4.5 cycles
                y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*sin(2*pi*f*t).*(1-alpha*((heaviside(t-t1)-heaviside(t-t2))));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
            
                t2=t1+0.06;               %3 cycles
                y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*sin(2*pi*f*t).*(1-alpha*((heaviside(t-t1)-heaviside(t-t2))));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                %}
            end
        end
    end
end
                          
figure(14)
plot(t,y)
title('Flicker+Sag');

for i=L:length(z)
    if rem(i,10)==0 
        X_test = vertcat(X_test,z(i,:));
        Y_test = vertcat(Y_test,cl(i));
    else
        X_train = vertcat(X_train,z(i,:));
        Y_train = vertcat(Y_train,cl(i));
    end
end
L=length(z)+1;

%% Flicker + Swell

x = "Flicker+Swell";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;

for alpha=0.1:0.077:0.8                          % Runs 10 times  
    for t1=0.04:0.0044444:0.058                      % Runs 5 times
        for alpha_flicker=0.08:0.02469135802:0.2          % Runs 5 times
            for beta=5:3.703703704:20                      % Runs 5 times          
                t2=t1+0.1;                %5 cycles
                y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*sin(2*pi*f*t).*(1+alpha*((heaviside(t-t1)-heaviside(t-t2))));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
 %{           
                t2=t1+0.09;               %4.5 cycles
                y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*sin(2*pi*f*t).*(1+alpha*((heaviside(t-t1)-heaviside(t-t2))));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
            
                t2=t1+0.06;               %3 cycles
                y = (1+alpha_flicker*sin(beta*2*pi*f*t)).*sin(2*pi*f*t).*(1+alpha*((heaviside(t-t1)-heaviside(t-t2))));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                %}
            end
        end
    end
end
                        
figure(15)
plot(t,y)
title('Flicker+Swell');

for i=L:length(z)
    if rem(i,10)==0 
        X_test = vertcat(X_test,z(i,:));
        Y_test = vertcat(Y_test,cl(i));
    else
        X_train = vertcat(X_train,z(i,:));
        Y_train = vertcat(Y_train,cl(i));
    end
end
L=length(z)+1;


%% Osciallatory + Sag

x = "Oscillatory Transient + Sag";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;

for alpha=0.1:0.0773480663:0.9                         % Runs 10 times
    for F_t=300:1160.493827:5000                    % Runs 5 times
        for t3=0.04:0.01:0.08                   % Runs 5 times
            for tau=0.008:0.007901234568:0.040          % Runs 5 times
                t4=t3+0.02;          % 1 cycle
                y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) * (alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
 %{               
                t4=t3+0.03;          % 1.5 cycles
                y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) *(alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                
                t4=t3+0.04;          %2 cycles
                y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) *(alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                %}
            end
        end 
    end
end
   
figure(16)
plot(t,y)
title('Oscillatory Transient + Sag');

for i=L:length(z)
    if rem(i,10)==0 
        X_test = vertcat(X_test,z(i,:));
        Y_test = vertcat(Y_test,cl(i));
    else
        X_train = vertcat(X_train,z(i,:));
        Y_train = vertcat(Y_train,cl(i));
    end
end
L=length(z)+1;


%% Osciallatory + Swell

x = "Oscillatory Transient + Swell ";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;

for alpha=0.1:0.0773480663:0.9                         % Runs 10 times
    for F_t=300:1160.493827:5000                    % Runs 5 times
        for t3=0.04:0.01:0.08                   % Runs 5 times
            for tau=0.008:0.007901234568:0.040          % Runs 5 times
                t4=t3+0.02;          % 1 cycle
                y= (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))) * (alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
 %{               
                t4=t3+0.03;          % 1.5 cycles
                y= (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                
                t4=t3+0.04;          %2 cycles
                y= (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                %}
            end
        end 
    end
end
   
figure(17)
plot(t,y)
title('Oscillatory Transient + Swell');

for i=L:length(z)
    if rem(i,10)==0 
        X_test = vertcat(X_test,z(i,:));
        Y_test = vertcat(Y_test,cl(i));
    else
        X_train = vertcat(X_train,z(i,:));
        Y_train = vertcat(Y_train,cl(i));
    end
end
L=length(z)+1;


%% Oscillatory Transients + Interruptions

x = "Oscillatory Transient + Interruptions";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;

for alpha=0.1:0.0773480663:0.9                         % Runs 10 times
    for F_t=300:1160.493827:5000                    % Runs 5 times
        for t3=0.04:0.01:0.08                   % Runs 5 times
            for tau=0.008:0.007901234568:0.040          % Runs 5 times
                t4=t3+0.02;          % 1 cycle
                y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) * (alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t));
                z= vertcat(z,y);
                cl=vertcat(cl,x);
 %{               
                t4=t3+0.03;          % 1.5 cycles
                y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                
                t4=t3+0.04;          %2 cycles
                y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                %}
            end
        end 
    end
end
   
figure(18)
plot(t,y)
title('Oscillatory Transient + Interruptions');

for i=L:length(z)
    if rem(i,10)==0 
        X_test = vertcat(X_test,z(i,:));
        Y_test = vertcat(Y_test,cl(i));
    else
        X_train = vertcat(X_train,z(i,:));
        Y_train = vertcat(Y_train,cl(i));
    end
end
L=length(z)+1;


%% Osciallatory + Harmonics 
% Ask Sir

x = "Oscillatory Transient + Harmonics";                            
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
f = 50;

for alpha=0.1:0.0773480663:0.9                         % Runs 10 times
    for F_t=300:1160.493827:5000                    % Runs 5 times
        for t3=0.04:0.01:0.08                   % Runs 5 times
            for tau=0.008:0.007901234568:0.040          % Runs 5 times
                for alpha3=0.05:0.002038735:0.15           % Runs 50 times
                    for alpha5=0.05:0.004158004158:0.15       % Runs 25 times
                t4=t3+0.02;          % 1 cycle
                y= (sin(2*pi*f*t)+ alpha*((heaviside(t-t1)-heaviside(t-t2))).exp(t1-t/tau).*sin(2*pi*F_t*t)) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)). 
                z= vertcat(z,y);
                cl=vertcat(cl,x);
 %{               
                t4=t3+0.03;          % 1.5 cycles
                y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                
                t4=t3+0.04;          %2 cycles
                y= (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))) + sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                %}
            end
        end 
    end
end
   
figure(19)
plot(t,y)
title('Oscillatory Transient + Harmonics');

for i=L:length(z)
    if rem(i,10)==0 
        X_test = vertcat(X_test,z(i,:));
        Y_test = vertcat(Y_test,cl(i));
    else
        X_train = vertcat(X_train,z(i,:));
        Y_train = vertcat(Y_train,cl(i));
    end
end
L=length(z)+1;


%% Inter-Harmonics

x = "Inter-Harmonics";                            
t = [0:t_s:0.2];
f=50;
%%ask sir about beta values
B1= 2;
B2= 5;
for alpha1=0.05:0.0100200401:0.15           % Runs 10 times
    for alpha2=0.05:0.0100200401:0.15       % Runs 10 times
        for alpha3=0.05:0.0200803213:0.15       % Runs 5 times
        y = alpha1*sin(2*pi*f*t)+ alpha2*sin(B1*2*pi*f*t)+ alpha3*sin(B2*2*pi*f*t);
        z= vertcat(z,y);
        cl=vertcat(cl,x);  
    end
end

figure(20)
plot(t,y)
title('Inter-Harmonics');

for i=L:length(z)
    if rem(i,10)==0 
        X_test = vertcat(X_test,z(i,:));
        Y_test = vertcat(Y_test,cl(i));
    else
        X_train = vertcat(X_train,z(i,:));
        Y_train = vertcat(Y_train,cl(i));
    end
end
L=length(z)+1;





