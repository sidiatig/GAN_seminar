function [ empirical_distribution,uniqueSamples,uniqueSize ] = SamplesToDistribution( samples )
%SAMPLESTODISTRIBUTION �˴���ʾ�йش˺�����ժҪ
%   �������е�����ͳ�ƾ���ֲ�
%   samples (1,N) NumOfSamples * 2

% support point
N = size(samples,2);  %% �ֲ��ĸ���
NumOfSamples = size(samples{1,1},1);  %% ÿ���ֲ��ĸ���

X = zeros(N*NumOfSamples,2);  % �������ʽ�洢���зֲ���������
for i = 1:N
    for j = 1:NumOfSamples
        X((i-1)*NumOfSamples+j,:)=samples{i}(j,:);
    end    
end

%����Լ��
%X(:,:) = round(X(:,:));
uniqueSamples = unique(X,'rows');
uniqueSize = size(uniqueSamples,1);
Count =zeros(1,uniqueSize);
for i = 1 : uniqueSize
    for j = 1 : N*NumOfSamples
        if uniqueSamples(i,:)==X(j,:)
            Count(1,i)=Count(1,i)+1;
        end
    end
end


empirical_distribution = (Count / sum(Count))';

end

