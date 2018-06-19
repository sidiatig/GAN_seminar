function [ w,tmp ] = SparseUpdateW( JointDistribution,mk,K,lambda,gamma,rho0 )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
w = zeros(1,K);

function [ s ] = SoftThreshold( k,a )
    if a > k
        s = a - k;
    elseif abs(a) < k
        s = 0;
    else
        s = a + k;
    end
end

tmp = sum(reshape(JointDistribution,[mk,K]));
a = tmp + lambda;

for idx = 1:K
    w(idx) = SoftThreshold(gamma/rho0,a(idx));
end

w = w / sum(w);

end

