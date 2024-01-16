function fitness=myfitness(pop)
global Image  I0 p    
A0=(pop/p).^2;
A0=A0';
Image_check=rescale(sum(A0.*I0,1));
Image1=rescale(Image);
A=sum(abs(Image_check-Image1).^2);
for i=1:20
    cha(i)=(pop(i+1)-pop(i))^2;
end
B=sum(cha);%增加限制条件防止过拟合（方法较为粗糙）
fitness=abs(A+0.0000005*B);
end