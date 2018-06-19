function [ w,tmp ] = update_w( JointDistribution,K,mk,lambda )
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   ����ϡ��Լ������w

tmp = sum(reshape(JointDistribution,[mk,K]));  % 1*K
w2  = tmp+lambda;  % 1*K
H = eye(K);
q = -  w2';
[w] = quadprog(H, q, [], [], ones(1,K), 1, zeros(K,1), [])';  % K*1
end

