function [] = fig_train_process_reg(info, fig_name, top)
    f = figure('Position',[200 200 900 225]);
    hold on
    yyaxis left
    c1 = [0.3 0.6 0.9 0.4];
    c3 = [0.9 0.4 0.15 0.4];
    c4 = [0.3 0.6 0.9];
    c2 = [0.35, 0.35, 0.35];
    if top
        h(1) = plot(NaN,NaN, '-','Color', c1);
        c1(4) = 0.2;
        plot(1:len(info.TrainingRMSE),info.TrainingRMSE, '-','Color', c1)
    else
        h(1) = plot(1:len(info.TrainingRMSE),info.TrainingRMSE, '-','Color', c1);
    end
    h(2) = plot(1:len(info.TrainingRMSE),smooth(info.TrainingRMSE,min(15,round(len(info.TrainingRMSE)*0.025))), '-','Color', c4);
    h(3) = plot(find(~isnan(info.ValidationRMSE)),info.ValidationRMSE(~isnan(info.ValidationRMSE)),...
        'Color',c2,'LineWidth',1,'LineStyle','--','Marker','o','MarkerFaceColor',c2,'MarkerSize',4);
    ylabel('Accuracy (%)');
    grid on;

    yyaxis right
    c5 = [0.9 0.4 0.15];
    if top
        h(4) = plot(NaN,NaN, '-','Color', c3);
        c3(4) = 0.2;
        plot(1:len(info.TrainingLoss),info.TrainingLoss,'-', 'Color', c3)
    else
        h(4) = plot(1:len(info.TrainingLoss),info.TrainingLoss,'-', 'Color', c3);
    end    
    h(5) = plot(1:len(info.TrainingLoss),smooth(info.TrainingLoss,min(15,round(len(info.TrainingLoss)*0.025))),'-', 'Color', c5);
    h(6) = plot(find(~isnan(info.ValidationLoss)),info.ValidationLoss(~isnan(info.ValidationLoss)),...
        'Color',c2,'LineWidth',1,'LineStyle',':','Marker','s','MarkerFaceColor',c2,'MarkerSize',4);
    xlim([-(len(info.TrainingRMSE)/100), len(info.TrainingRMSE)+(len(info.TrainingRMSE)/100)])
    ylabel('Loss');

    if top
        legend(h, {'Training RMSE','Smoothed Training RMSE','Validation RMSE','Training Loss','Smoothed Training Loss','Validation Loss'},'Location','east')
    else
        set(gca,'XAxisLocation','top');
        xlabel('Iteration');
    end
    set(gca, 'GridAlpha', 0.35);
    set(gca, 'GridColor', [0, 0, 0]);
    set(gcf, 'color', 'white');
    set(gca, 'color', 'white');
    

    export_fig(fig_name, '-q101', '-transparent', f);
    
    close all;
    
end