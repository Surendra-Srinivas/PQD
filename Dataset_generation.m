%% PQD GENERATION (NORMAL + 19 PQDS) 
% 1000 SIGNALS FOR EACH PQD WITH 640 SAMPLE POINTS (1000 signals/class /SNR)
% TRAIN DATA = 70%, VALIDATION = 20% TEST DATA = 10%

clc
clear all

pi=3.141592654;
t_s = 1/3200;
t1_s = 0.00074844075;
z=[]; cl=[]; X_train=[]; Y_train=[]; X_test=[]; Y_test=[];

%% Normal
x = "Normal";
t = [0: ts :0.2-ts];                   % 640 sample points per disturbance
for f = 49.5:0.02020:50.5    % 1250 different sine waves  (Runs 50 times)
    for vm=0.97:0.0401606:1.03 % (Runs 25 times)
    y = vm*sin(2*pi*f*t);
    y =awgn(y, SNR);
    z= vertcat(z,y);
    cl=vertcat(cl,x);
    end
end

figure(1)
plot(t,y)
title('Pure Sine wave')

for i=1:length(z)
    if rem(i,10)==0 
        X_test = vertcat(X_test,z(i,:));
        Y_test = vertcat(Y_test,cl(i));
    else
        X_train = vertcat(X_train,z(i,:));
        Y_train = vertcat(Y_train,cl(i));
    end
end
L=length(z)+1;

%% Sag

x = "Sag";                            
t = [0:ts:0.2];
f = 50;

for alpha=0.1:0.0163:0.85              % Runs 50 times
    for t1=0.04:t1_s:0.058         % Runs 25 times
        y=(1- alpha*((heaviside(t-t1)-heaviside(t-(t1+0.1))))).*sin(2*pi*f*t);  %5 cycles
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

figure(2)
plot(t,y);
title('Sag disturbance');

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

%% Swell

x = "Swell";                            
t = [0:ts:0.2];
f = 50;

for alpha=0.1:0.01428:0.8              % Runs 50 times
    for t1=0.04:t1_s:0.058          % Runs 25 times
        y=(1+ alpha*((heaviside(t-t1)-heaviside(t-(t1+0.1))))).*sin(2*pi*f*t);
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

figure(3)
plot(t,y);
title('Swell disturbance');

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

%% Interruption

x = "Interruption";                            
t = [0:ts:0.2-ts];
f = 50;

for alpha=0.9:0.00204:1                 % Runs 50 times
    for t1=0.04:t1_s:0.058           % Runs 25 times
        y=(1-alpha*((heaviside(t-t1)-heaviside(t-(t1+0.1))))).*sin(2*pi*f*t);
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

figure(4)
plot(t,y);
title('Interruption');

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

%% Harmonics

x = "Harmonics";                            
t = [0:t_s:0.2];


for alpha3=0.05:0.040032026:0.15           % Runs 25 times
    for alpha5=0.05:0.005005005:0.15       % Runs 20 times
        for f=49.9:0.1:50                        % Runs 2 times
            alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
            y = alpha1*sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*pi*f*t);
            z= vertcat(z,y);
            cl=vertcat(cl,x); 
        end
    end
end

figure(5)
plot(t,y)
title('Harmonics');

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

%% Flicker

x = "Flicker";                            
t = [0:t_s:0.2];

for alpha_flicker=0.06:0.007007007:0.2    % Runs 20 times
    for beta=8:0.6805444355:25              % Runs 25 times
        for f=49.9:0.1:50                        % Runs 2 times
            y=(1+alpha_flicker*sin(beta*2*pi*f*t));
            z= vertcat(z,y);
            cl=vertcat(cl,x);
        end
    end
end
        
figure(6)
plot(t,y)
title('Flicker');

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

%% Oscillatory Transient

x = "Oscillatory Transient";                            
t = [0:t_s:0.2];
f = 50;

for alpha=0.1:0.0773480663:0.8                         % Runs 10 times
    for F_t=300:1160.493827:5000                    % Runs 5 times
        for t3=0.04:0.01:0.08                   % Runs 5 times
            for tau=0.008:0.007901234568:0.040          % Runs 5 times
                t4=t3+0.02;          % 1 cycle
                y= sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                z= vertcat(z,y);
                cl=vertcat(cl,x);
 %{               
                t4=t3+0.03;          % 1.5 cycles
                y= sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                
                t4=t3+0.04;          %2 cycles
                y= sin(2*pi*f*t)+ alpha*(heaviside(t-t3)-heaviside(t-t4)).*exp(t3-t/tau).*sin(2*pi*F_t*t);
                z= vertcat(z,y);
                cl=vertcat(cl,x);
                %}
            end
        end 
    end
end
   
figure(7)
plot(t,y)
title('Oscillatory Transient');

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

%% Notch

x = "Notch";                            
t = [0:t_s:0.2];
f = 50;

for alpha=0.1:0.0006116207951:0.4;            % Runs 50 times
    for t1=0.001:0.00037422037642:0.01;        % Runs 25 times
        t2=t1+0.0005;      % 0.025 cycle
        sum = 0;
        for n=0:9
            sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
        end
        y = sin(2*pi*f*t) - alpha*sign(2*pi*f*t).*sum;
        z= vertcat(z,y);
        cl=vertcat(cl,x);
%{        
        t2=t1+0.001;       % 0.05 cycle
        sum = 0;
        for n=0:9
            sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
        end
        y = sin(2*pi*f*t) - alpha*sign(2*pi*f*t).*sum;
        z= vertcat(z,y);
        cl=vertcat(cl,x);
    %}
    end
end

figure(8)
plot(t,y)
title('Notch');

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

%% Spike

x = "Spike";                            
t = [0:t_s:0.2];
f = 50;

for alpha=0.1:0.0006116207951:0.4;            % Runs 50 times
    for t1=0.001:0.00037422037642:0.01;        % Runs 25 times
        t2=t1+0.0005;      % 0.025 cycle
        sum = 0;
        for n=0:9
            sum = sum + ( heaviside(t-(t1+0.02*n))-heaviside(t-(t2+0.02*n)) );
        end
        y = sin(2*pi*f*t) + alpha*sign(2*pi*f*t).*sum;
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

figure(9)
plot(t,y)
title('Spike');

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

%% Sag + Harmonics

x = "Sag+Harmonics";                            
t = [0:t_s:0.2];
f = 50;

for alpha=0.1:0.0888:0.9                         % Runs 10 times  
    for t1=0.04:0.0044444:0.058                    % Runs 5 times
        for alpha3=0.05:0.025:0.15               % Runs 5 times
                for alpha5=0.05:0.02469135802:0.15       % Runs 5 times
                    alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
            
                    t2=t1+0.1;                %5 cycles
                    y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
%{            
                    t2=t1+0.09;               %4.5 cycles
                    y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
            
                    t2=t1+0.06;               %3 cycles
                    y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
                    %}
                end
        end
    end
end

figure(10)
plot(t,y)
title('Sag+Harmonics');

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

%% Swell + Harmonics

x = "Swell+Harmonics";                            
t = [0:0.0003129:0.2];
f = 50;

for alpha=0.1:0.0777:0.8                         % Runs 10 times  
    for t1=0.04:0.0044444:0.058                    % Runs 5 times
        for alpha3=0.05:0.025:0.15               % Runs 5 times
                for alpha5=0.05:0.02469135802:0.15       % Runs 5 times
                    alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
            
                    t2=t1+0.1;                %5 cycles
                    y = (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
%{            
                    t2=t1+0.09;               %4.5 cycles
                    y = (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
            
                    t2=t1+0.06;               %3 cycles
                    y = (1+alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
                    %}
                end
        end
    end
end

figure(11)
plot(t,y)
title('Swell+Harmonics');

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

%% Interruption + Harmonics
%% same formula as sag+harmonic

x = "Interruption+Harmonics";                            
t = [0:t_s:0.2];
f = 50;

for alpha=0.9:0.0111:1                           % Runs 10 times  
    for t1=0.04:0.0044444:0.058                    % Runs 5 times
        for alpha3=0.05:0.025:0.15               % Runs 5 times
                for alpha5=0.05:0.02469135802:0.15       % Runs 5 times
                    alpha1 = sqrt(1 - alpha3^2 - alpha5^2);
            
                    t2=t1+0.1;                %5 cycles
                    y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
%{            
                    t2=t1+0.09;               %4.5 cycles
                    y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
            
                    t2=t1+0.06;               %3 cycles
                    y = (1-alpha*((heaviside(t-t1)-heaviside(t-t2)))).*(alpha1* sin(2*pi*f*t)+ alpha3*sin(3*2*pi*f*t)+ alpha5*sin(5*2*pi*f*t));
                    z= vertcat(z,y);
                    cl=vertcat(cl,x);
                    %}
                end
        end
    end
end


figure(12)
plot(t,y)
title('Interruption+Harmonics');

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

%% Flicker + Harmonics

x = "Flicker+Harmonics";                            
t = [0:t_s:0.2];
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
t = [0:0.0003129:0.2];
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
t = [0:0.0003129:0.2];
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
t = [0:t_s:0.2-t_s];
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
t = [0:t_s:0.2-t_s];
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
t = [0:t_s:0.2-t_s];
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
t = [0:t_s:0.2-t_s];
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





