clc;clear;close
% 
w = 2*pi;
alpha = 1.2;
% A = [w*(alpha-1) 0; 0 -w]
% B = [1;1]
% C = [1 1]
% D = 0
% 
g1 = 1.2699;
% G = [g1 0];
% disp(eig(A));
% disp(eig(A-B*G));
% 
% sys1 = ss(A,B,C,D);
% sys2 = ss(A-B*G,B,C,D);
end_point = 10
step_size = 0.01;
t_val= 0:step_size:end_point;
syms t
A11(t) = w*(alpha-1) + g1*sin(w*t)*(cos(w*t)-sin(w*t))
A12 = 0
A21(t) = g1*cos(w*t)*(cos(w*t)-sin(w*t))
A22 = -w
A(t) = [A11(t) A12;A21(t) A22]
B1(t) = -sin(w*t)
B2(t) = cos(w*t)
B(t) =[ B1(t);B2(t)]

C(t) = [cos(w*t) sin(w*t)];
D = 0;


z1=zeros(1,length(t_val));
z2=zeros(1,length(t_val));
yp=zeros(1,length(t_val));
r=ones(1,length(t_val));



for i = 2:length(t_val)
    z1(i) = z1(i-1)+ step_size*(A11(t_val(i-1))*z1(i-1) + A12*z2(i-1) + B1(t_val(i-1))*r(i-1));
    z2(i) = z2(i-1)+ step_size*(A21(t_val(i-1))*z1(i-1) + A22*z2(i-1) + B2(t_val(i-1))*r(i-1));

    yp(i) = C(t_val(i))*[z1(i);z2(i)];
    
end

plot(t_val,yp)
xlim([0,end_point])
title('Step Response of Time-Varying System with g = 1.2699')
legend('Step Response')
xlabel('Time')
ylabel('Amplitude')
savefig('Floquet_Theory_1_Unstable')