function [ X ] = UpdateX( PI,w,Samples,K )
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   X Ϊ�����Ժ�����ʵļ�Ȩ��
%   X K*2
X = zeros(K,2);
for i = 1:K
    X(i,:) = sum(repmat(PI(i,:)',[1,2]).*Samples)/w(i); % PI(i,:)/w(i) �������Ժ�����ʵļ�Ȩ��
end

end

