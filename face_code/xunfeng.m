clear
clc
%% 寻找区域最大值点
path = 'E:\\dataset\\FAS\\face0604-0607\\face0604\\0607ji\\';
name = '1.tif';
Imagezong=imread([path, name]);
I=im2gray(Imagezong);
I=rescale(I);
D=zeros(5120,5120);
for i=31:5090
    for j=31:5090
        if max(max(I(i-20:i+20,j-20:j+20)))==I(i,j) && I(i,j)~=0
            Imagezong(i,j,1)=0;
            Imagezong(i,j,3)=0;
            D(i,j)=1;
            Imagezong(i,j,2)=255;
        end
    end
end
%% 排除相邻的一些点
for i=31:5090
    for j=31:5090
        if D(i,j)==1
            test=D(i-20:i+20,j-20:j+20);
            test(21,21)=0;
            if max(max(test))==1
                D(i,j)=0;
                Imagezong(i,j,1)=0;
                Imagezong(i,j,3)=0;
                Imagezong(i,j,2)=100;
            end
        end
    end
end

figure, imshow(Imagezong), hold on

%% 确定每个区域的具体位置
t=0;
for i=31:5090
    for j=31:5090

        if D(i,j)==1
            t=t+1;
            yup(t)=i-30;
            ydown(t)=i+30;
            xzuo(t)=j-30;
            xyou(t)=j+30;

            if t == 4800
                disp(i);
                disp(j);
%             end
%             if mod(t,100) == 0
                plot([i,i+1], [j, j+1], 'Color','r','LineWidth',1);
                plot([xzuo(t), xyou(t)], [yup(t), yup(t)], 'Color','g','LineWidth',1);
                plot([xzuo(t), xyou(t)], [ydown(t), ydown(t)], 'Color','r','LineWidth',1);
                plot([xzuo(t), xzuo(t)], [yup(t), ydown(t)], 'Color','b','LineWidth',1);
                plot([xyou(t), xyou(t)], [yup(t), ydown(t)], 'Color','w','LineWidth',1);
            end
        end
    end
end


%% create 光源
yuan = ones(1.0,51);
guiyi = ones(1.0,51);

%% save
save('540-590yuan.mat', 'yuan');
save('540-590guiyi.mat', 'guiyi');
save('xzuo.mat', 'xzuo');
save('xyou.mat', 'xyou');
save('yup.mat', 'yup');
save('ydown.mat', 'ydown');



