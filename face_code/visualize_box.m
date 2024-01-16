%% visualize bounding boxs computed by xunfeng.m

path = 'E:\\dataset\\FAS\\face0604-0607\\face0604\\0607ji\\';
name = '1.tif';
Imagezong=imread([path, name]);
figure, imshow(Imagezong), title(name);
% figure, imshow(bishijie_00001'), title(name);
hold on

%% draw boxes on the base image

% for t = floor(sum(D(:))/2) : 1 : floor(sum(D(:))/2) + 1000
% 
%     if xzuo(t) > 2000 && xzuo(t) < 3000 ...
%             && yup(t) > 2000  && yup(t) < 3000
%         plot([xzuo(t), xyou(t)], [yup(t), yup(t)], 'Color','g','LineWidth',1);
%         plot([xzuo(t), xyou(t)], [ydown(t), ydown(t)], 'Color','r','LineWidth',1);
%         plot([xzuo(t), xzuo(t)], [yup(t), ydown(t)], 'Color','b','LineWidth',1);
%         plot([xyou(t), xyou(t)], [yup(t), ydown(t)], 'Color','w','LineWidth',1);
%     end
%     
% %     t = t + 1;
% %     plot([xzuo(t), xyou(t)], [yup(t), yup(t)], 'Color','g','LineWidth',1);
% %     plot([xzuo(t), xyou(t)], [ydown(t), ydown(t)], 'Color','r','LineWidth',1);
% %     plot([xzuo(t), xzuo(t)], [yup(t), ydown(t)], 'Color','b','LineWidth',1);
% %     plot([xyou(t), xyou(t)], [yup(t), ydown(t)], 'Color','w','LineWidth',1);
%     
% end
% plot([2563,2564], [2347, 2348], 'Color','r','LineWidth',30);

%% draw 51 point in the target box

x1 = 2461;
y1 = 2451;
r1 = 30;
xzuo1 = x1 - r1;
xyou1 = x1 + r1;
yup1 = y1 - r1;
ydown1 = y1 + r1;

plot([xzuo1, xyou1], [yup1, yup1], 'Color','g','LineWidth',1);
plot([xzuo1, xyou1], [ydown1, ydown1], 'Color','r','LineWidth',1);
plot([xzuo1, xzuo1], [yup1, ydown1], 'Color','b','LineWidth',1);
plot([xyou1, xyou1], [yup1, ydown1], 'Color','w','LineWidth',1);

colors = [ [1., 0., 0.]; [1., 0.5, 0.]; [1., 1., 0.]; [0., 1., 0.]; [0., 1., 1.]; [0., 0., 1.]; [0.5, 0., 1.] ];
markers = [ 'o', '+', '*', '.', 'x', '_', '|', 's'];

[rows, cols, ~] = size(Imagezong);
for num = 1 : 1 : 51
    load(['./D/', 'D', '_', num2str(num), '.mat']);
%     D = D';
    for row = yup1 : ydown1
        for col = xzuo1 : xyou1
            if D(row, col) == 1
                ratio = 1.0 * (51-num/2) / 51.0;
                color = ratio * colors(floor(num/ 8) + 1, :);
                marker = markers(mod(num , 7) + 1);
                plot(col, row, 'Marker', marker, 'MarkerFaceColor', color, 'MarkerEdgeColor', color);
%                 switch mod(num,3)
%                     case 1
%                         plot(col, row, 'Marker', 'o', 'MarkerFaceColor', [ 1.0 * (51-num/2) / 51.0, 0, 0], ...
%                             'MarkerEdgeColor', [1.0 * (51-num/2) / 51.0, 0, 0]);
%                     case 2
%                         plot(col, row, 'Marker', 'o', 'MarkerFaceColor', [ 0, 1.0 * (51-num/2) / 51.0, 0], ...
%                             'MarkerEdgeColor', [0, 1.0 * (51-num/2) / 51.0, 0]);
%                     case 0
%                         plot(col, row, 'Marker', 'o', 'MarkerFaceColor', [ 0, 0, 1.0 * (51-num/2) / 51.0], ...
%                             'MarkerEdgeColor', [0, 0, 1.0 * (51-num/2) / 51.0]);
%                 end
                if mod(num , 7) == 0
                    yup=row-30;
                    ydown=row+30;
                    xzuo=col-30;
                    xyou=col+30;
                color = ratio * colors(floor(num/ 8) + 1, :);
                    plot([xzuo, xyou], [yup, yup], 'Color',color,'LineWidth',1);
                    plot([xzuo, xyou], [ydown, ydown], 'Color',color,'LineWidth',1);
                    plot([xzuo, xzuo], [yup, ydown], 'Color',color,'LineWidth',1);
                    plot([xyou, xyou], [yup, ydown], 'Color',color,'LineWidth',1);
                end
            end
        end
    end
end


