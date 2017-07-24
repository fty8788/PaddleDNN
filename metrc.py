from sklearn import metrics

label = []
y = []

with open("paddle_gu_1_10.balance.s.t.p") as f:
    for line in f:
        fs = line.split(' ')
        if len(fs) < 802:
            continue
        label.append(float(fs[800]))
        y.append(float(fs[802]))
fpr, tpr, thresholds = metrics.roc_curve(label, y, pos_label=1)
print metrics.auc(fpr, tpr)
