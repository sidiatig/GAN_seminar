function [ output_Samples ] = GenerateSamples( N,NumOfSamples,mu,covariance )
%GENERATESAMPLES �˴���ʾ�йش˺�����ժҪ
%  N ��ά��˹�ֲ�����
%  NumOfSamples ÿ����ά��˹�ֲ�����������
%  mu  2*N ��˹�ֲ���ֵ����
%  covariance ��˹�ֲ�Э�������

if N~=size(mu,2)
    fprintf('��˹�ֲ�������ƥ��');
    return;
end
output_Samples = cell(1,N);
for i = 1:N
    output_Samples{1,i} = mvnrnd(mu{i},covariance{i},NumOfSamples);
end

