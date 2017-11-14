import csv
import numpy as np
churn_dat = []
log_dat = []
churn_limit = 100000
log_limit = 100000
msno_to_churn = {}
churn_to_msno = {}
msno_to_logs = {}
logs_to_msno = {}
logs_to_churn = {}
msnos = []

with open('datasets/train.csv/train.csv', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile, fieldnames = ["msno", "is_churn"])
    count = 0
    for row in reader:
        msno = row["msno"]
        is_churn = row["is_churn"]
        churn_dat.append([msno, is_churn])
        msno_to_churn[msno] = is_churn
        churn_to_msno[is_churn] = msno
        msnos.append(msno)
        count += 1
        if count == churn_limit:
            break
#print(churn_dat[:10])
with open('datasets/user_logs.csv/user_logs.csv', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile,
        fieldnames = ["msno", "date", "num_25","num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"])
    count = 0
    for row in reader:
        msno = row["msno"]
        num_25 = row["num_25"]
        num_50 = row["num_50"]
        num_75 = row["num_75"]
        num_985 = row["num_985"]
        num_100 = row["num_100"]
        apnd = [msno, num_25, num_50, num_75, num_985, num_100]
        msno_to_logs[msno] = apnd[1:]
        log_dat.append(apnd)
        count += 1
        if count == log_limit:
            break
X_train_list = []
Y_train_list = []
for i in msnos:
    try:
        logs = msno_to_logs[i]
        churn = msno_to_churn[i]
        X_train_list.append(logs)
        Y_train_list.append(churn)
    except KeyError:
        pass
X_train_list = [list(map(int, x)) for x in X_train_list[1:]]
Y_train_list = list(map(int, Y_train_list[1:]))
X_train = np.asarray(X_train_list)
Y_train = np.asarray(Y_train_list)
indices = np.array([i for i in range(len(X_train))])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
print(X_train)
print(Y_train)
print(X_train.shape)
print(Y_train.shape)