function [ s ] = SoftThreshold( k,a )
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   soft thresholding operator

if a > k
    s = a - k;
elseif abs(a) < k
    s = 0;
else
    s = a + k;
end

end

