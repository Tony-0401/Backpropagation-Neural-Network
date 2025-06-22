clc;
clear all;
close all;

% 生成訓練和測試資料
t = linspace(-0.8, 0.7, 400);
t1 = t+(rand(1, 400)-0.5)*0.01;
t2 = t+(rand(1, 400)-0.5)*0.01;

randnx1 = t1(randperm(400));
randnx2 = t2(randperm(400));

Xtrain = randnx1(1:300);
Ytrain = randnx2(1:300);
Xtest = randnx1(301:400);
Ytest = randnx2(301:400);

% 訓練和測試的目標值
Ztrain = 5*sin(pi*Xtrain.^2).*sin(2*pi*Ytrain)+1;
Ztest = 5*sin(pi*Xtest.^2).*sin(2*pi*Ytest)+1;

% 正規化
ymin = min([Ztrain Ztest]);
ymax = max([Ztrain Ztest]);
Dtrain = (Ztrain-ymin)/(ymax-ymin)*(0.8-0.2)+0.2;
Dtest = (Ztest-ymin)/(ymax-ymin)*(0.8-0.2)+0.2;


% 設定參數
n = 20; % 隱藏層神經元數量
learn = 0.8; % 學習率
a = 0.7; % 動量係數
N = 40000; % 訓練次數

% 隨機初始化bias和權重
range = 4;
start = -2;

bias = rand(1, n)*range+start;
outputW = rand(1, n)*range+start;
outputbias = rand(1, 1)*range+start;
W1 = rand(1, n)*range+start;
W2 = rand(1, n)*range+start;

% 訓練過程
for t = 1:N
    % 隱藏層的輸出
    hiddenV = W1'*Xtrain+W2'*Ytrain+bias'*ones(1, 300);
    hiddenY = 1./(1+exp(-hiddenV));
    
    % 輸出層的輸出
    outputV = outputW*hiddenY+outputbias*ones(1, 300);
    outputY = 1./(1+exp(-outputV));
    
    % 計算誤差
    e = Dtrain-outputY;
    Etrain = (1/2)*(e.^2);
    
    % 計算local gradient
    og = e.*outputY.*(1-outputY);
    hg = hiddenY.*(1-hiddenY).*(outputW'*og);
    
    % 更新權重和bias
    for i = 1:300
        if i == 1
            deltabias(:, 1) = learn*hg(:, 1)*1;
            deltaW1(:, 1) = learn*hg(:, 1)*Xtrain(1);
            deltaW2(:, 1) = learn*hg(:, 1)*Ytrain(1);
            deltaoutputW(:, 1) = learn*og(:, 1)*hiddenY(:, 1);
            deltaoutputbias(:, 1) = learn*og(:, 1)*1;
        else
            deltabias(:, i) = learn*hg(:, i)*1+a*deltabias(:, i-1);
            deltaW1(:, i) = learn*hg(:, i)*Xtrain(i)+a*deltaW1(:, i-1);
            deltaW2(:, i) = learn*hg(:, i)*Ytrain(i)+a*deltaW2(:, i-1);
            deltaoutputW(:, i) = learn*og(:, i)*hiddenY(:, i)+a*deltaoutputW(:, i-1);
            deltaoutputbias(:, i) = learn*og(:, i)*1+a*deltaoutputbias(:, i-1);
        end
    end
    
    bias = bias+(sum(deltabias'))/300;
    W1 = W1+(sum(deltaW1'))/300;
    W2 = W2+(sum(deltaW2'))/300;
    outputW = outputW+(sum(deltaoutputW'))/300;
    outputbias = outputbias+(sum(deltaoutputbias'))/300;
    Eavg(t) = (sum(Etrain))/300;
end

Etrain_avg = Eavg(N);

% 測試資料的輸入
testhiddenV = W1'*Xtest+W2'*Ytest+bias'*ones(1, 100);
testhiddenZ = 1./(1+exp(-testhiddenV));
testoutputV = outputW*testhiddenZ+outputbias*ones(1, 100);
testoutputZ = 1./(1+exp(-testoutputV));

% 計算測試誤差
testError = Dtest-testoutputZ;
Etest = 0.5*(testError.^2);
Etest_avg = mean(Etest);   

% 反量化
realY = (outputY-0.2)/0.6*(ymax-ymin)+ymin;
Dtrain = (Dtrain-0.2)/0.6*(ymax-ymin)+ymin;
testoutputZ = (testoutputZ-0.2)/0.6*(ymax-ymin)+ymin;

% 繪製圖形
figure(1)
plot(1:t, Eavg);
title('Error-training');
xlabel('Training times');
ylabel('Error');

figure(2)
subplot(222);
xline1 = linspace(min(Xtrain), max(Xtrain), 50);
yline1 = linspace(min(Ytrain), max(Ytrain), 50);
[XX, YY] = meshgrid(xline1, yline1);
ZZ = griddata(Xtrain, Ytrain, Dtrain, XX, YY, 'cubic');
mesh(XX, YY, ZZ);
title('期望輸出');

subplot(223);
xline2 = linspace(min(Xtrain), max(Xtrain), 50);
yline2 = linspace(min(Ytrain), max(Ytrain), 50);
[XX2, YY2] = meshgrid(xline2, yline2);
ZZ2 = griddata(Xtrain, Ytrain, realY, XX2, YY2, 'cubic');
mesh(XX2, YY2, ZZ2);
title('訓練輸出');

subplot(224);
xline3 = linspace(min(Xtest), max(Xtest), 50);
yline3 = linspace(min(Ytest), max(Ytest), 50);
[XX3, YY3] = meshgrid(xline3, yline3);
ZZ3 = griddata(Xtest, Ytest, testoutputZ, XX3, YY3, 'cubic');
mesh(XX3, YY3, ZZ3);
title('測試輸出');
