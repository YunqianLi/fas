clear 
clc
%%
global p I0 Image
load('540-590yuan.mat')
load('540-590guiyi.mat')
load('xzuo.mat')
load('xyou.mat')
load('ydown.mat')
load('yup.mat')
% load('channel.mat')
% load('white.mat')

%% 
n=2;%%光谱分辨率
    L=1:n:51;
    m=length(L);


%% 获取基
path = 'E:\\dataset\\FAS\\face0604\\基\\';
filelist = dir(path);
for i1=1:m
    Imagezong=imread([path, filelist(2+L(i1)).name]);
    Ia=im2gray(Imagezong);
    Ia=single(Ia);

    Imagezong=imread([path, 'jibj_00001.tif']);
    Ib=im2gray(Imagezong);
    Ib=single(Ib);
    
    I_sumab=abs(Ia-Ib);
    
    I_zong(:,:,i1)=rescale(I_sumab./guiyi(1,i1));
end


%% 获得每个通道的数据
for t=1:9314
    ji(:,:,:,t)=I_zong(yup(t):ydown(t),xzuo(t):xyou(t),:);
end
c=sum(ji,2); % 
channel(:,:,:)=c(:,1,:,:);
% c111 = zeros(size(c11),'single');
% for i = 1 : 60
%     c111 = c111 + ji(1,i,:,:);
% end

%% 获得白光通道结果
Imagebai=imread('fake_00002.tif');
Imagebai=im2gray(Imagebai);

I_baia=single(Imagebai);

Imagebai=imread('fake_00003.tif');
Imagebai=im2gray(Imagebai);
I_baib=single(Imagebai);


I_b=abs(I_baia-I_baib);%%%%%去掉背景光后的结果

for t=1:9314
    b(:,:,t)=I_b(yup(t):ydown(t), xzuo(t):xyou(t));
end
bai=sum(b,2);
white(:,:)=bai(:,1,:);

%%

h=waitbar(0,'努力搬砖中');
for t=1:9314%图中共有9314个像素点

    I0=rand(61,m);%%二维
    Im=double(white);
    Image=Im(:,t);%%二维
    Image=Image';
    I0(:,:)=channel(:,:,t);
    I0=I0';
    I0=I0./max(max(I0));

    gridnum=m;
    newpop(:,:,t)=zeros(1,gridnum);
    p=100;
    for i=1:15

        pop=ones(1,gridnum);



        options = optimoptions('ga','MaxTime',3600*48,'MaxStallGenerations',10000,'MaxGenerations',80,...
            'ConstraintTolerance',0.000001,'InitialPopulationMatrix',pop, 'PlotFcn',@gaplotbestf);
        [pop,fval]=ga(@myfitness,gridnum,[],[],[],[],zeros(1,gridnum),9*p*ones(1,gridnum),[],1:gridnum,options);


        newpop(:,:,t)=newpop(:,:,t)+pop;
    end
    str=['搬砖中~~~',num2str(t/9314*100),'%'];
    waitbar(t/1600,h,str)

end
%最终获得newpop即为每个像素点的最后光谱重建结果。目前由于matlab中GA算法包不能并行运算，所以计算速度较慢。
%%
% lamda=540:590;
% for i=1:51
%     fanshe(i)=newpop(i)/yuan(i);
% end
% figure
% plot(lamda,smooth(smooth((fanshe))))