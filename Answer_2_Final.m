load realitymining.mat
%house 1, work 2, elsewhere 3, nosig  0, off NaN
M=s(4).data_mat';
 %% Call Function
Mbw=generate_binary(M');
%% Answer 2A
% Biplots for explaining the relation between the data points and the first 3 eigen vectors.
%%
[coeff,score,latent,tsquared,explained] = pca(Mbw);

for i=1:120
    labels{i} =num2str(i);
end
figure
biplot(coeff(:,1:3),'scores',score(10,1:3),'varlabels', labels)
xlabel('Component_1')
ylabel('Component_2')
zlabel('Component_3')
title('Day 10')
%%
figure
biplot(coeff(:,1:3),'scores',score(15,1:3),'varlabels', labels)
xlabel('Component_1')
ylabel('Component_2')
zlabel('Component_3')
title('Day 15')
%%
figure
biplot(coeff(:,1:3),'scores',score(20,1:3),'varlabels', labels)
xlabel('Component_1')
ylabel('Component_2')
zlabel('Component_3')
title('Day 20')
%% Eigenvectors
figure
numb=1;
ax1=subplot(3,2,1);
plot(coeff(1:24,numb),'ko-')
hold on
plot(Mbw(10, 1:24),'o-')
plot(Mbw(15, 1:24),'b-')
plot(Mbw(20, 1:24),'g-')

legend('Eigvec 1','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' home'];
title(title_name)

ax2=subplot(3,2,2);
plot(coeff(25:48,numb),'ko-')
hold on
plot(Mbw(10, 25:48),'o-')
plot(Mbw(15, 25:48),'b-')
plot(Mbw(20, 25:48),'g-')

legend('Eigvec 1','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' work'];
title(title_name)

ax3=subplot(3,2,3);
plot(coeff(49:72,numb),'ko-')
hold on
plot(Mbw(10, 49:72),'o-')
plot(Mbw(15, 49:72),'b-')
plot(Mbw(20, 49:72),'g-')

legend('Eigvec 1','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' elsewhere'];
title(title_name)

ax4=subplot(3,2,4);
plot(coeff(73:96,numb),'ko-')
hold on
plot(Mbw(10, 73:96),'o-')
plot(Mbw(15, 73:96),'b-')
plot(Mbw(20, 73:96),'g-')

legend('Eigvec 1','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' no signal'];
title(title_name)

ax5=subplot(3,2,5);
plot(coeff(97:120,numb),'ko-')
hold on
plot(Mbw(10, 97:120),'o-')
plot(Mbw(15, 97:120),'b-')
plot(Mbw(20, 97:120),'g-')

legend('Eigvec 1','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' off'];
title(title_name)
ylim([min(coeff(:,numb)),max(coeff(:,numb))])
linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')
%%
figure
numb=2;
ax1=subplot(3,2,1);
plot(coeff(1:24,numb),'ko-')
hold on
plot(Mbw(10, 1:24),'o-')
plot(Mbw(15, 1:24),'b-')
plot(Mbw(20, 1:24),'g-')

legend('Eigvec 2','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' home'];
title(title_name)

ax2=subplot(3,2,2);
plot(coeff(25:48,numb),'ko-')
hold on
plot(Mbw(10, 25:48),'o-')
plot(Mbw(15, 25:48),'b-')
plot(Mbw(20, 25:48),'g-')

legend('Eigvec 2','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' work'];
title(title_name)

ax3=subplot(3,2,3);
plot(coeff(49:72,numb),'ko-')
hold on
plot(Mbw(10, 49:72),'o-')
plot(Mbw(15, 49:72),'b-')
plot(Mbw(20, 49:72),'g-')

legend('Eigvec 2','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' elsewhere'];
title(title_name)

ax4=subplot(3,2,4);
plot(coeff(73:96,numb),'ko-')
hold on
plot(Mbw(10, 73:96),'o-')
plot(Mbw(15, 73:96),'b-')
plot(Mbw(20, 73:96),'g-')

legend('Eigvec 2','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' no signal'];
title(title_name)

ax5=subplot(3,2,5);
plot(coeff(97:120,numb),'ko-')
hold on
plot(Mbw(10, 97:120),'o-')
plot(Mbw(15, 97:120),'b-')
plot(Mbw(20, 97:120),'g-')

legend('Eigvec 2','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' off'];
title(title_name)
ylim([min(coeff(:,numb)),max(coeff(:,numb))])
linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')
%%
figure
numb=3;
ax1=subplot(3,2,1);
plot(coeff(1:24,numb),'ko-')
hold on
plot(Mbw(10, 1:24),'o-')
plot(Mbw(15, 1:24),'b-')
plot(Mbw(20, 1:24),'g-')

legend('Eigvec 3','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' home'];
title(title_name)

ax2=subplot(3,2,2);
plot(coeff(25:48,numb),'ko-')
hold on
plot(Mbw(10, 25:48),'o-')
plot(Mbw(15, 25:48),'b-')
plot(Mbw(20, 25:48),'g-')

legend('Eigvec 3','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' work'];
title(title_name)

ax3=subplot(3,2,3);
plot(coeff(49:72,numb),'ko-')
hold on
plot(Mbw(10, 49:72),'o-')
plot(Mbw(15, 49:72),'b-')
plot(Mbw(20, 49:72),'g-')

legend('Eigvec 3','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' elsewhere'];
title(title_name)

ax4=subplot(3,2,4);
plot(coeff(73:96,numb),'ko-')
hold on
plot(Mbw(10, 73:96),'o-')
plot(Mbw(15, 73:96),'b-')
plot(Mbw(20, 73:96),'g-')

legend('Eigvec 3','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' no signal'];
title(title_name)

ax5=subplot(3,2,5);
plot(coeff(97:120,numb),'ko-')
hold on
plot(Mbw(10, 97:120),'o-')
plot(Mbw(15, 97:120),'b-')
plot(Mbw(20, 97:120),'g-')

legend('Eigvec 3','day 10','day 15','day 20')
grid on
title_name = ['Eigvec ',num2str(numb),' off'];
title(title_name)
ylim([min(coeff(:,numb)),max(coeff(:,numb))])
linkaxes([ax1,ax2,ax3,ax4,ax5],'xy')
%% Answer 2_B
% Reconstructed for days 10,15 and 20.
%%
[coeff,score,latent,tsquared,explained] = pca(Mbw);
day=10
Psi=mean(Mbw,1)
Vsample=Mbw(day,:)
rc=zeros(1,size(Vsample,2));
sample=3
for i=1:sample
    proj=(Vsample-Psi)*coeff(:,i);
    rc=rc+proj*coeff(:,i)';
    %proj_vec(i)=proj
end
rc=rc+Psi;

figure
plot(rc,'o-')
ylim([min(rc) max(rc)]);
xlabel('Hours')
ylabel('Values')
legend('Reconstructed day 10')
title('Reconstructed day 10')
%%
[coeff,score,latent,tsquared,explained] = pca(Mbw);
day=15
Psi=mean(Mbw,1)
Vsample=Mbw(day,:)
rc=zeros(1,size(Vsample,2));
sample=3
for i=1:sample
    proj=(Vsample-Psi)*coeff(:,i);
    rc=rc+proj*coeff(:,i)';
    rcc=rc+Psi
    %proj_vec(i)=proj
end
rc=rc+Psi;
figure
plot(rc,'o-')
ylim([min(rc) max(rc)]);
xlabel('Hours')
ylabel('Values')
legend('Reconstructed day 15')
title('Reconstructed day 15')
%%
[coeff,score,latent,tsquared,explained] = pca(Mbw);
day=20
Psi=mean(Mbw,1)
Vsample=Mbw(day,:)
rc=zeros(1,size(Vsample,2));
sample=3
for i=1:sample
    proj=(Vsample-Psi)*coeff(:,i);
    rc=rc+proj*coeff(:,i)';
    %proj_vec(i)=proj
end
rc=rc+Psi;
figure
plot(rc,'o-')
ylim([min(rc) max(rc)]);
xlabel('Hours')
ylabel('Values')
legend('Reconstructed day 20')
title('Reconstructed day 20')
%% Answer 2_C
figure
pareto(explained)
xlabel('Principal component')
ylabel('Explained(%)')
gname
%%
[coeff,score,latent,tsquared,explained] = pca(Mbw);
day=10
Psi=mean(Mbw,1)
Vsample=Mbw(day,:)
rc=zeros(1,size(Vsample,2));
sample=24
for i=1:sample
    proj=(Vsample-Psi)*coeff(:,i);
    rc=rc+proj*coeff(:,i)';
end
rc=rc+Psi;
error=norm(Mbw(day,:)-rc)/norm(Mbw(day,:));
Accuracy=1-error;
disp(Accuracy)
%%
[coeff,score,latent,tsquared,explained] = pca(Mbw);
day=15
Psi=mean(Mbw,1)
Vsample=Mbw(day,:)
rc=zeros(1,size(Vsample,2));
sample=33
for i=1:sample
    proj=(Vsample-Psi)*coeff(:,i);
    proj
    rc=rc+proj*coeff(:,i)';
end
rc=rc+Psi;
error=norm(Mbw(day,:)-rc)/norm(Mbw(day,:));
Accuracy=1-error;
disp(Accuracy)
%%
[coeff,score,latent,tsquared,explained] = pca(Mbw);
day=20
Psi=mean(Mbw,1)
Vsample=Mbw(day,:)
rc=zeros(1,size(Vsample,2));
sample=42
for i=1:sample
    proj=(Vsample-Psi)*coeff(:,i);
    rc=rc+proj*coeff(:,i)';
end
rc=rc+Psi;
error=norm(Mbw(day,:)-rc)/norm(Mbw(day,:));
Accuracy=1-error;
disp(Accuracy)
%% Answer 2_D
[coeff,score,latent,tsquared,explained] = pca(Mbw);
Psi=mean(Mbw,1)
sample=120
Vsample=Mbw(1,:)
rc=zeros(1,size(Vsample,2));
Accuracy_list=zeros(1,size(Vsample,2));

for i=1:size(Accuracy_list,2);
    proj1=(Mbw(i,:)-Psi)*coeff(:,1);
    proj2=(Mbw(i,:)-Psi)*coeff(:,2);
    proj3=(Mbw(i,:)-Psi)*coeff(:,3);
    rc2=rc+proj1*coeff(:,1)'+proj2*coeff(:,2)'+proj3*coeff(:,3)'+Psi;
    error=norm(Mbw(i,:)-rc2)/norm(Mbw(i,:));
    Accuracy_list(i)=1-error;
end
Accuracy_list
Min_acc_I=find(Accuracy_list==min(Accuracy_list(:)));
Min_acc=Accuracy_list(1,Min_acc_I);
fprintf('The day with the minimum accuracy is: day %d, \n And its accuracy is %d',Min_acc_I,Min_acc)
%% Answer 2_E
%%
figure
plot(score(:,1),score(:,2),'+')
xlabel('1st principal component')
ylabel('2nd principal component')
gname
%% Answer 2_F
tsquared
size(tsquared)
size(M)
%%
[st,index]=sort(tsquared,'descend')
extreme=index(1:5);
extreme
%%
% Here we plot the Days changing day
day=extreme(1);
DAY=Mbw(day,:);
figure
ax1=subplot(3,2,1);
plot(DAY(1:24),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' home'];
title(title_name)

ax2=subplot(3,2,2);
plot(DAY(25:48),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' work'];
title(title_name)

ax3=subplot(3,2,3);
plot(DAY(49:72),'ro-')
grid on
title_name = ['Vlues Day ',num2str(day),' elsewhere'];
title(title_name)

ax4=subplot(3,2,4);
plot(DAY(73:96),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' no signal'];
title(title_name)

ax5=subplot(3,2,5);
plot(DAY(97:120),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' off'];
title(title_name)

day=extreme(2);
DAY=Mbw(day,:);
figure
ax1=subplot(3,2,1);
plot(DAY(1:24),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' home'];
title(title_name)

ax2=subplot(3,2,2);
plot(DAY(25:48),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' work'];
title(title_name)

ax3=subplot(3,2,3);
plot(DAY(49:72),'ro-')
grid on
title_name = ['Vlues Day ',num2str(day),' elsewhere'];
title(title_name)

ax4=subplot(3,2,4);
plot(DAY(73:96),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' no signal'];
title(title_name)

ax5=subplot(3,2,5);
plot(DAY(97:120),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' off'];
title(title_name)

day=extreme(3);
DAY=Mbw(day,:);
figure
ax1=subplot(3,2,1);
plot(DAY(1:24),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' home'];
title(title_name)

ax2=subplot(3,2,2);
plot(DAY(25:48),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' work'];
title(title_name)

ax3=subplot(3,2,3);
plot(DAY(49:72),'ro-')
grid on
title_name = ['Vlues Day ',num2str(day),' elsewhere'];
title(title_name)

ax4=subplot(3,2,4);
plot(DAY(73:96),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' no signal'];
title(title_name)

ax5=subplot(3,2,5);
plot(DAY(97:120),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' off'];
title(title_name)

day=extreme(4);;
DAY=Mbw(day,:);
figure
ax1=subplot(3,2,1);
plot(DAY(1:24),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' home'];
title(title_name)

ax2=subplot(3,2,2);
plot(DAY(25:48),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' work'];
title(title_name)

ax3=subplot(3,2,3);
plot(DAY(49:72),'ro-')
grid on
title_name = ['Vlues Day ',num2str(day),' elsewhere'];
title(title_name)

ax4=subplot(3,2,4);
plot(DAY(73:96),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' no signal'];
title(title_name)

ax5=subplot(3,2,5);
plot(DAY(97:120),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' off'];
title(title_name)

day=extreme(5);
DAY=Mbw(day,:);
figure
ax1=subplot(3,2,1);
plot(DAY(1:24),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' home'];
title(title_name)

ax2=subplot(3,2,2);
plot(DAY(25:48),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' work'];
title(title_name)

ax3=subplot(3,2,3);
plot(DAY(49:72),'ro-')
grid on
title_name = ['Vlues Day ',num2str(day),' elsewhere'];
title(title_name)

ax4=subplot(3,2,4);
plot(DAY(73:96),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' no signal'];
title(title_name)

ax5=subplot(3,2,5);
plot(DAY(97:120),'ro-')
grid on
title_name = ['Values Day ',num2str(day),' off'];
title(title_name)
%%
