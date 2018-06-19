%% ���ɶ�ά��˹�ֲ�
clc;
clear;
N = 5;
NumOfSamples = 40;
K = 10;
mu = cell(1,N);
covariance = cell(1,N);
%% ָ��ÿ���ֲ��ľ�ֵ������Э�������
mu{1,1} = [2,1]; mu{1,2} = [-1,10];mu{1,3} = [-7,-10];mu{1,4} = [1,-8];mu{1,5}=[-5,0];
covariance{1,1} = [1,0;0,1];covariance{1,2} = [1,0;0,1];covariance{1,3} = [1,0;0,1];
covariance{1,4} = [1,0;0,1];covariance{1,5} = [1,0;0,1];

%% ȷ�����ɵ�ÿ����������һ��
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
% title('��������ֲ�');
%saveas(gcf,['./','����ֲ�_',num2str(N),'.jpg']);

%% ��ʼ�����Ϸֲ�
PI = rand(K,mk);
PI = PI ./ sum(sum(PI)) ;
% image(PI,'CDataMapping','scaled');
% colorbar;
%% ��ʼ��w,x
w = rand(K,1);
w = w / sum(w);

% lagrangians multiplier
lambda = zeros(1,K);
% prepare block matrix
diag_matrix = repmat({ones(mk)},1,K); 

vecsize = K*mk;

%% ���Ϸֲ���������������������
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
% x ���������� 10^-5 ����ѭ��
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
    %% ��ѭ��������Ϸֲ�
    % �������µĳ�ʼ�׶� Ȩ�غ����Ϸֲ��ĸ��������Ƚϴ�
    % ��support λ�ø��¼��κ�
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
disp(['����ʱ��: ',num2str(toc)]);

for i = 1:K
   if w(i) ~=0
       plot(x(i,1),x(i,2),'b+');
       hold on;
   end
end

title('��˹���������ֲ�');
saveas(gcf,['./','��˹���������ֲ�_',num2str(N),'.jpg']);  
figure();
image(PI,'CDataMapping','scaled');
colorbar;















