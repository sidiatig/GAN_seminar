%% 生成二维高斯分布
clc;
clear;
N = 5;
NumOfSamples = 40;
K = 10;
mu = cell(1,N);
covariance = cell(1,N);
%% 指定每个分布的均值向量和协方差矩阵
mu{1,1} = [2,1]; mu{1,2} = [-1,10];mu{1,3} = [-7,-10];mu{1,4} = [1,-8];mu{1,5}=[-5,0];
covariance{1,1} = [1,0;0,1];covariance{1,2} = [1,0;0,1];covariance{1,3} = [1,0;0,1];
covariance{1,4} = [1,0;0,1];covariance{1,5} = [1,0;0,1];

%% 确保生成的每个样本都不一样
while 1
    samples = GenerateSamples(N,NumOfSamples,mu,covariance);
    [emp_distribution,Samples,mk] = SamplesToDistribution(samples);
    if mk == N * NumOfSamples
        break;
    end
end
%% plot
for i = 1:N
    plot(samples{i}(:,1),samples{i}(:,2),'r+');
    hold on;
end
% figure();
% stem(emp_distribution);
% title('样本经验分布');
%saveas(gcf,['./','经验分布_',num2str(N),'.jpg']);

%% 初始化联合分布
PI = rand(K,mk);
PI = PI ./ sum(sum(PI)) ;
% image(PI,'CDataMapping','scaled');
% colorbar;
%% 初始化w,x
w = rand(K,1);
w = w / sum(w);

% lagrangians multiplier
lambda = zeros(1,K);
% prepare block matrix
diag_matrix = repmat({ones(mk)},1,K); 

vecsize = K*mk;

%% 联合分布矩阵向量化，按行排列
JointDistribution = reshape(PI',[vecsize,1]);

%% rho
rho0 = 50;
nIter = 10;
Tadmm = 10;
updated_x = zeros(1,nIter);  
updated_w = zeros(nIter,Tadmm);
updated_pi= zeros(nIter,Tadmm);
Initial_x = rand(K,2);
x = zeros(K,2);
gamma = 0.7;
theta = 0;
tic;
% x 更新量大于 10^-5 保持循环
% while updated_x > 1e-8 
for i = 1:nIter
    % with fixed w and PI,update x
%     x = theta * Initial_x + (1-theta) * UpdateX(PI,w,Samples,K);
    for idx = 1:K
        if w(idx) ~= 0
            xtmp = sum(repmat(PI(idx,:)',[1,2]).*Samples)/w(idx);
            x(idx,:) = theta * Initial_x(idx,:) + (1-theta) * xtmp;
        else
            x(idx,:) = Initial_x(idx,:);
        end
    end
    updated_x(i) = norm(x - Initial_x ,2);
    Initial_x = x;
    % update cost matrix  K*mk
    D = CostMatrix(x,Samples,K,mk);
    % update rho
    % rho = rho0*sum(sum(D))/(vecsize);
    %% 内循环求解联合分布
    % 迭代更新的初始阶段 权重和联合分布的更新量都比较大
    % 在support 位置更新几次后
    for j = 1:Tadmm
        % update w
        init_w = w;
%       [w,tmp] = update_w(JointDistribution,K,mk,lambda);
        [ w,tmp ] = SparseUpdateW( JointDistribution,mk,K,lambda,gamma,rho0 );
        updated_w(i,j) = norm(w-init_w,2);
        
        % iterative update joint_distribution
        H = 2*rho0*blkdiag(diag_matrix{:,:});
        q = Prepare_q( lambda,w,rho0,D,K,mk );
        Aeq = repmat(eye(mk),[1,K]); % mk*mk*K
        beq = emp_distribution;       % mk*1
        init_J = JointDistribution;
        [JointDistribution] = quadprog(H, q, [], [], Aeq, beq, zeros(vecsize,1), []);
        updated_pi(i,j) = norm(JointDistribution - init_J,2);
        
        % update lambda
        lambda = lambda + tmp -w;
    end
    PI = reshape(JointDistribution,[mk,K])';  % K*mk
end
toc;
disp(['运行时间: ',num2str(toc)]);

for i = 1:K
   if w(i) ~=0
       plot(x(i,1),x(i,2),'b+');
       hold on;
   end
end

title('高斯采样样本分布');
saveas(gcf,['./','高斯采样样本分布_',num2str(N),'.jpg']);  
figure();
image(PI,'CDataMapping','scaled');
colorbar;















