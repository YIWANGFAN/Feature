# -*- coding: utf-8 -*-
# @Time    : 2021/10/13 16:13
# @Author  : wxf
# @FileName: NBclassify.py
# @Software: PyCharm
# @Email ：15735952634@163.com
import numpy as np
import torch
import pdb
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier as DNN
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import csv
import os
from sklearn.model_selection import train_test_split
def get_data():
    no_pre_train=[]
    no_pre_test=[]
    no_dnf_floder = ('/Users/wxf/PycharmProjects/feature/densenet/wxf')
    # dnf_floder = ('F:\pythonProject\DNF\DNF\\alexnet')
    no_dnf = os.listdir(no_dnf_floder)
    # print(no_dnf)
    for i in range(len(no_dnf)):
        if 'train' in no_dnf[i]:
            if 'no' not in no_dnf[i]:
            # if 'no' in no_dnf[i]:
                no_pre_train.append(no_dnf[i])
    print(no_pre_train)
    for i in range(len(no_dnf)):
        if 'test' in no_dnf[i]:
            if 'no' not in no_dnf[i]:
            # if 'no' in no_dnf[i]:                         # 想找预训练后的直接写not in
                no_pre_test.append(no_dnf[i])
    for i in range(len(no_pre_train)):
        for j in range(len(no_pre_test)):
            if no_pre_train[i].split('_')[0] == no_pre_test[j].split('_')[0]:
                # print(no_pre_test[j])
                # print(no_pre_train[i])
                data = np.load(no_dnf_floder +'/'+ no_pre_train[i])
                data1 = np.load(no_dnf_floder + '/' + no_pre_test[j])
                lst = data.files
                for item in lst:
                    print(item)
                x_vector = data['vector']
                label_vector = data['utt']
                x_vector1 = data1['vector']
                label_vector1 = data1['utt']
                x_vector = np.r_[x_vector,x_vector1]
                label_vector = np.r_[label_vector,label_vector1]
                accsum = 0
                recallsum = 0
                f1sum = 0
                gmeansum = 0
                for a in range (0,10):
                    print(a)
                    X_train, X_test, y_train, y_test = train_test_split(x_vector, label_vector, test_size=0.3, random_state=0)
                    clf = SVC()
                    clf.fit(X_train,y_train)
                    pre = clf.predict(X_test)
                    acc = sklearn.metrics.accuracy_score(y_test, pre)
                    # print(acc)
                    recall = sklearn.metrics.recall_score(y_test, pre,average='macro')
                    f1 = sklearn.metrics.f1_score(y_test, pre,average='macro')
                    tn, fp, fn, tp = confusion_matrix(y_test, pre,labels=[0,36]).ravel()
                    print(tn,fp)
                    specificity = tn / (tn + fp)
                    gmean = pow(recall*specificity,1/2)
                    accsum = accsum+acc
                    recallsum = recallsum+recall
                    f1sum = f1sum+f1
                    gmeansum = gmeansum+gmean
                accave = accsum/10
                recallave = recallsum/10
                f1ave = f1sum/10
                gmeanave = gmeansum/10
                print(accave,recallave,f1ave,gmeanave)

                # print(no_pre_train[i].split('_'))
                path = './scoreBySVM.csv'

                with open(path, "a+", newline='\n') as file:  # 处理csv读写时不同换行符  linux:\n    windows:\r或者\n    mac:\r
                    csv_file = csv.writer(file)
                    datas = [[no_pre_train[i].split('_')[0] + '_' +no_pre_train[i].split('_')[3].split('.')[0], gmeanave,accave,recallave,f1ave]]
                    # datas = [[no_pre_train[i].split('_')[0] + '_' + no_pre_train[i].split('_')[2] + '_' + no_pre_train[i].split('_')[5].split('.')[0], gmeanave,accave,recallave,f1ave]]
                    csv_file.writerows(datas)


get_data()








