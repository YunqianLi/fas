import os
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

# 将Matplotlib切换到非交互模式
plt.ion()

path_fake = 'data/test/TEST/FAKE/59/'
path_true = 'data/test/TEST/TRUE/59/'

fakeList = [f for f in os.listdir(path_fake) if '.npy' in f]
prob_fake = np.load(path_fake + fakeList[0])
for idx, file in enumerate(fakeList):
    if idx > 0:
        prob_fake = np.append(prob_fake, np.load(path_fake + file))
print('prob_fake is ', prob_fake)

# 减少 fake 数量
# prob_fake = prob_fake[0::2]
# print('prob_fake(part) is ', prob_fake)

trueList = [f for f in os.listdir(path_true) if '.npy' in f]
prob_true = np.load(path_true + trueList[0])
for idx, file in enumerate(trueList):
    if idx > 0:
        prob_true = np.append(prob_true, np.load(path_true + file))
print('prob_true is ', prob_true)

len_fake = len(prob_fake)
len_true = len(prob_true)
print('len_fake is ', len_fake)
print('len_true is ', len_true)

label_fake = np.zeros(len_fake)
label_true = np.ones(len_true)
label_all = np.concatenate((label_fake, label_true))
scores_all = np.concatenate((prob_fake, prob_true))
print('label_all is ', label_all)
print('scores_all is ', scores_all)

# 假设y_true是真实标签，y_scores是模型的预测概率
y_true = label_all
y_scores = scores_all
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算AUC
roc_auc = auc(fpr, tpr)
print('roc_auc is ', roc_auc)

# 最佳阈值
# best_threshold_location = np.argmax(tpr - fpr)
# best_threshold = thresholds[best_threshold_location]
best_threshold = 0.5
print('best_threshold is ', best_threshold)

# 计算 confusion matrix
scores_all_binary = np.where(scores_all > best_threshold, 1, 0)
print('scores_all_binary is ', scores_all_binary)
cm = confusion_matrix(label_all, scores_all_binary)
print(cm)
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]
NEG = cm[0, 0] + cm[0, 1]
POS = cm[1, 0] + cm[1, 1]
ALL = NEG + POS
ACC = (TN + TP) / ALL * 100
FAR = FP / NEG * 100
FRR = FN / POS * 100
print('ACC is {:.2f}%'.format(ACC))
print('FAR is {:.2f}%'.format(FAR))
print('FRR is {:.2f}%'.format(FRR))

# 绘制ROC曲线
plt.figure(1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:0.2f}%)'.format(roc_auc * 100))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 绘制 confusion matrix
plt.figure(2)
# 定义类标签
class_labels = ['Negative', 'Positive']

# 创建热图
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

# 添加颜色刻度条
plt.colorbar()

# 设置坐标轴刻度标签
plt.xticks(np.arange(len(class_labels)), class_labels)
plt.yticks(np.arange(len(class_labels)), class_labels)

# 添加数值标签
thresh = cm.max() / 2.0
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, str(cm[i, j]) + '\n' + format(100 * cm[i, j] / ALL, '.2f') + '%', ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

# 添加标签
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix \n (Threshold={:.3f} ACC={:.2f}% FAR={:.2f}% FRR={:.2f}%)'.format(best_threshold, ACC, FAR, FRR))

# 显示图形
plt.show(block=True)
