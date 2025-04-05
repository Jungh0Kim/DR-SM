
clear; close all;
load motorcycle

rand_idx = randperm(131,100);
X = X(rand_idx,:);
y = y(rand_idx,:);

[X, Idd] = sort(X);
for ii=1:length(Idd)
    y(ii,1) = y(Idd(ii),1);
end

[NMSE, NLPD, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
    vhgpr_ui(X, y, X, y,100);

figure()
plotvarianza(X, mutst, diagSigmatst)
hold on
plot(X, mutst,'k','Linewidth',1.5)
xlim([min(X) max(X)])
set(gcf,'color','w')
xlabel('x data','fontname','Arial','FontSize',13)
ylabel('noise g(\it{x})','fontname','Arial','FontSize',13)
hold off

figure()
plotvarianza(X, atst, diagCtst); hold on;
scatter(X, y,30,'x','MarkerEdgeColor','b','LineWidth',1.0);
plot(X, atst,'k','Linewidth',1.5)
xlim([min(X) max(X)])
set(gcf,'color','w')
xlabel('x data','fontname','Arial','FontSize',13)
ylabel('latent function f(\it{x})','fontname','Arial','FontSize',13)
hold off


figure()
plotvarianza(X, Ey, Vy); hold on
plot(X, Ey,'k','Linewidth',1.5);
scatter(X, y, 35,'x','MarkerEdgeColor','k','LineWidth',1.1);
xlim([min(X) max(X)])
xlabel('Random variable, {\itx}','fontname','Arial','FontSize',12.5)
ylabel('Predictions by heteroscedastic GP model','fontname','Arial','FontSize',12.5)
lgnd = legend(' Predictive variance',' Predictive mean',' Heteroscedastic noisy observation');
set(lgnd,'FontName','Arial','FontSize',12,'NumColumns',1); legend boxoff;
set(gcf,'color','w')
hold off

