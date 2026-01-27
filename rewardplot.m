clear all
close all
clc
x=1:1:140001;
load('Reward1.mat')
average = movmean(A2Creward, 1000);
average2 = movmean(IMreward, 1000);

% ====== Read PPO & SAC from CSV (current folder) ======
ppoTbl = readtable('PPO_cumulative_reward.csv');   % columns: episode, cumulative_reward
sacTbl = readtable('SAC_cumulative_reward.csv');   % columns: episode, cumulative_reward

PPOreward = ppoTbl{:,2};
SACreward = sacTbl{:,2};

xPPO = ppoTbl{:,1} + 1;  % 你原来 episode 从 1 开始画，所以 +1
xSAC = sacTbl{:,1} + 1;

% smooth window: keep your style, but avoid window > length
wPPO = min(1000, max(1, floor(numel(PPOreward)/10)));
wSAC = min(1000, max(1, floor(numel(SACreward)/10)));

average3 = movmean(PPOreward, wPPO);
average4 = movmean(SACreward, wSAC);

% ====== Define consistent colors once ======
C.A2C = slanCL(1651);  colA2C = C.A2C(5,:);  fillA2C = C.A2C(3,:);
C.PPO = slanCL(1648);  colPPO = C.PPO(5,:);  fillPPO = C.PPO(1,:);
C.IM  = slanCL(1647);  colIM  = C.IM(5,:);   fillIM  = C.IM(3,:);
C.SAC = slanCL(1655);  colSAC = C.SAC(5,:);  fillSAC = C.SAC(3,:);

figure

% ==================== Top subplot ====================
subplot(2,1,1)

plot(x,   average,  'Color', colA2C, 'Linewidth', 2); hold on
plot(xPPO,average3, 'Color', colPPO, 'Linewidth', 2);
plot(x,   average2, 'Color', colIM,  'Linewidth', 2);
plot(xSAC,average4, 'Color', colSAC, 'Linewidth', 2);

% --- A2C envelope patch (keep your findpeaks style) ---
envelope_up = A2Creward(:);
envelope_down = -A2Creward(:);
Location1 = x(:);
Location2 = x(:);
for i = 1:3
    [top, location_up] = findpeaks(envelope_up);
    [bottom, location_down] = findpeaks(envelope_down);
    envelope_up = top(:);
    envelope_down = bottom(:);
    Location1 = Location1(location_up);
    Location2 = Location2(location_down);
end
n = min([numel(Location1), numel(Location2), numel(envelope_up), numel(envelope_down)]);
Xenv = [Location1(1:n); flipud(Location2(1:n))];
Yenv = [envelope_up(1:n); flipud(-envelope_down(1:n))];
patch(Xenv, Yenv, fillA2C, 'FaceAlpha', 0.5, 'edgealpha', 0);

% --- IM envelope patch ---
envelope_up = IMreward(:);
envelope_down = -IMreward(:);
Location1 = x(:);
Location2 = x(:);
for i = 1:3
    [top, location_up] = findpeaks(envelope_up);
    [bottom, location_down] = findpeaks(envelope_down);
    envelope_up = top(:);
    envelope_down = bottom(:);
    Location1 = Location1(location_up);
    Location2 = Location2(location_down);
end
n = min([numel(Location1), numel(Location2), numel(envelope_up), numel(envelope_down)]);
Xenv = [Location1(1:n); flipud(Location2(1:n))];
Yenv = [envelope_up(1:n); flipud(-envelope_down(1:n))];
patch(Xenv, Yenv, fillIM, 'FaceAlpha', 0.5, 'edgealpha', 0);

% --- SAC: transparent raw line (RGBA) ---
alphaLine = 0.1;
plot(xSAC, SACreward, 'Color', [colSAC alphaLine], 'LineWidth', 1.5);

% --- PPO envelope patch (findpeaks style) ---
envelope_up = PPOreward(:);
envelope_down = -PPOreward(:);
Location1 = xPPO(:);
Location2 = xPPO(:);
for i = 1:3
    [top, location_up] = findpeaks(envelope_up);
    [bottom, location_down] = findpeaks(envelope_down);
    envelope_up = top(:);
    envelope_down = bottom(:);
    Location1 = Location1(location_up);
    Location2 = Location2(location_down);
end
n = min([numel(Location1), numel(Location2), numel(envelope_up), numel(envelope_down)]);
Xenv = [Location1(1:n); flipud(Location2(1:n))];
Yenv = [envelope_up(1:n); flipud(-envelope_down(1:n))];
patch(Xenv, Yenv, fillPPO, 'FaceAlpha', 0.5, 'edgealpha', 0);

text(45437, IMreward(45437)+25, 'Teacher replace by agent 5', ...
    'FontName', 'Times New Roman', 'FontSize', 14);

set(gca, 'FontName', 'Times New Roman','FontSize', 14);
ylabel('Culmulative reward','Fontweight','bold','FontSize', 14)
xlabel('Episode','Fontweight','bold','FontSize', 14)
set(gca,'YLim',[-100 40]);
grid on
legend('Average A2C','Average PPO','Average IM+DRL','Average SAC')

% ==================== Bottom subplot ====================
subplot(2,1,2)
load('data_Monte2.mat')   % data_A2C,data_IM,data_PPO,data_SAC

D = [data_A2C(:), data_PPO(:), data_IM(:), data_SAC(:)];
pos = 1:4;

box_h = boxplot(D, ...
    'Colors', [0 0 0], ...
    'Symbol', '', ...
    'Positions', pos, ...
    'Widths', 0.55);
set(box_h, 'LineWidth', 1.2);
hold on

boxobj = findobj(gca, 'Tag', 'Box'); % order usually right-to-left
fillColors = [fillA2C; fillPPO; fillIM; fillSAC];

for i = 1:numel(boxobj)
    k = numel(boxobj) - i + 1; % map to A2C,PPO,IM,SAC
    patch(get(boxobj(i), 'XData'), get(boxobj(i), 'YData'), fillColors(k,:), ...
        'FaceAlpha', 0.5, 'LineWidth', 1.1);
end

set(gca, 'XLim', [0.5 4.5]);
xticks(pos)
xticklabels({'A2C','PPO','IM+DRL','SAC'})
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);
ylabel('Performance', 'Fontweight', 'bold', 'FontSize', 14)
xlabel('Algorithm',  'Fontweight', 'bold', 'FontSize', 14)
grid on

set(gcf,'color','w');

