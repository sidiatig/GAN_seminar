function [ D ] = CostMatrix( x,Samples,K,mk )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �������ĵ������ľ������
%   L2
D = zeros(K,mk);
for i = 1:K
    for j = 1:mk
        D(i,j) = norm(x(i,:)-Samples(j,:),2);
    end
end
end

