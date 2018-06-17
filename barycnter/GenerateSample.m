%% 生成二维高斯分布
clc;
clear;
N = 5;
NumOfSamples = 40;
mu = cell(1,N);
covariance = cell(1,N);
%% 指定每个分布的均值向量和协方差矩阵
mu{1,1} = [2,1]; mu{1,2} = [10,4];mu{1,3} = [-8,-2];mu{1,4} = [4,-8];mu{1,5}=[-5,8];
covariance{1,1} = [1,0;0,1];covariance{1,2} = [1,0;0,1];covariance{1,3} = [1,0;0,1];
covariance{1,4} = [1,0;0,1];covariance{1,5} = [1,0;0,1];
Samples = GenerateSamples(N,NumOfSamples,mu,covariance);


% 根据样本生成经验分布
[emp_distribution,X,mk] = SamplesToDistribution(Samples);

% figure();
% stem(emp_distribution);
% title('样本经验分布');
%saveas(gcf,['./','经验分布_',num2str(N),'.jpg']);

%% 初始化联合分布
PI = rand(N,mk);
PI = PI ./ sum(sum(PI)) ;
% image(PI,'CDataMapping','scaled');
% colorbar;
%% 初始化w,x
w = rand(1,N);
w = w / sum(w);
x = zeros(2,N);  % 2*N

% lagrangians multiplier
lambda = ones(1,N)/N;

% with fixed w and PI,update x
for i = 1:N
    x(:,i) = sum(repmat(PI(i,:)',[1,2]).*X)/w(i);
end

%% plot
% for i = 1:N
%     plot(Samples{i}(:,1),Samples{i}(:,2),'r+');
%     hold on;
% end
% 
% plot(x(1,:),x(2,:),'b+');
% title('高斯采样样本分布');
%saveas(gcf,['./','高斯采样样本分布_',num2str(N),'.jpg']);  

















