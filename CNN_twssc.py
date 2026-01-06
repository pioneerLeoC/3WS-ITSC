# -*- coding = utf-8 -*-
# @Time : 2021/12/11 10:27
# @Author : Z_C_Lee
# @File : RNN_twssc.py
# @Software : PyCharm


import pandas as pd
import torch
from torchtext.legacy import data
from sampling.sampleToCSV import extractByFile_random, extractByFile_fisvdd
from tools.processCSV import find_lack_label
import numpy as np
import time
import os
from main.nn.TextCNN import TextCNN

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HIP_LAUNCH_BLOCKING'] = '1'
class TWSSC:
    def __init__(self, trainFolder, validFolder, sampleFolder, all_train_filename, all_test_filename):
        self.alpha=None
        self.beta=None
        self.model=None
        self.TP=0
        self.FN=0
        self.FP=0
        self.TN=0
        self.current_lack_len=0
        self.current_all_correct=0
        self.current_lack_correct=0

        self.trainFolder=trainFolder
        self.validFolder=validFolder
        self.sampleFolder=sampleFolder

        self.all_train_filename=all_train_filename
        self.all_test_filename=all_test_filename

        self.train_file_all_num=None

    def split_valid(self, filename):
        head, tail = os.path.split(filename)

        file=pd.read_csv(filename, header=None)
        train = file.sample(frac=0.9, random_state=1, axis=0)
        valid = file[~file.index.isin(train.index)]

        train = train.sort_values(by=1, ascending=False)
        train.to_csv('{0}{1}'.format(head+"/main_train/",tail), header=False, encoding="utf-8", sep=",",
                     index=False)
        valid.to_csv('{0}{1}'.format(head+"/valid_files/",tail), header=False, encoding="utf-8", sep=",",
                     index=False)


    def compute_lamb(self, num_file):
        """
        计算代价矩阵
        :param num_file:
        :return:
        """
        # up = 20 / self.train_file_all_num
        # lamb_PN = 60
        # lamb_NP = 40
        # lamb_BP = num_file * up
        # lamb_BN = lamb_BP

        # up = 10 / self.train_file_all_num
        # lamb_PN = 60 - (num_file - 1) * up
        # lamb_NP = 40 - (num_file - 1) * up
        # lamb_BP = num_file * up
        # lamb_BN = lamb_BP

        self.lamb_PN = 750
        self.lamb_NP = 150
        self.lamb_BP = 40
        self.lamb_BN = 40

        return 0,  self.lamb_PN,  self.lamb_BP,  self.lamb_BN,  self.lamb_NP, 0

    def compute_threshold(self, num_file):
        """
        计算阈值
        :param all_file_len:
        :param num_file:
        :return:
        """
        lamb_PP, lamb_PN, lamb_BP, lamb_BN, lamb_NP, lamb_NN = self.compute_lamb(num_file)
        self.alpha = (lamb_PN - lamb_BN) / ((lamb_PN - lamb_BN) + (lamb_BP - lamb_PP))
        self.beta = (lamb_BN - lamb_NN) / ((lamb_BN - lamb_NN) + (lamb_NP - lamb_BP))

    def compute_entropy(self, prob_arr):
        value = 0
        for prob in prob_arr:
            value += -prob * np.log2(prob)
        return value

    def one_file_train(self, optimizer, criterion, train_iter, valid_iter, epochs):
        """
        单个文件训练及测试
        :param optimizer: 优化器
        :param criterion: 损失函数
        :param train_iter: 训练集迭代器
        :param valid_iter: 验证集迭代器
        :param epochs: 训练次数
        :return:
        """
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for epoch in range(1, epochs + 1):
            train_loss = 0.0
            self.model.train()
            for index, batch in enumerate(train_iter):
                batch.label = (batch.label - 1) * (-1)
                y = batch.label.to(DEVICE)
                x = batch.review.to(DEVICE)
                outputs = self.model(x)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item() * batch.review.size(0)  # 累计每一批次损失
            train_loss /= len(train_iter)
            print("Epoch:{},Train_Loss:{}".format(epoch, train_loss))

            self.model.eval()
            correct = 0
            valid_loss = 0.0
            all_len = 0
            with torch.no_grad():  # 不进行梯度计算
                for index, batch in enumerate(valid_iter):
                    batch.label = (batch.label - 1) * (-1)
                    # context = batch.review.to(DEVICE)
                    # target = batch.label.to(DEVICE)
                    y = batch.label.to(DEVICE)
                    x = batch.review.to(DEVICE)
                    outputs = self.model(x)
                    loss = criterion(outputs, y)
                    valid_loss += loss.data.item() * batch.review.size(0)  # 累计每一批次损失
                    predict = outputs.argmax(1)

                    correct += predict.eq(batch.label.view_as(predict)).sum().item()
                    all_len += batch.review.size(1)
                    correct_ratio = correct / (all_len)

                valid_loss /= len(valid_iter)
                print("Epoch:{},Valid_Loss:{}".format(epoch, valid_loss))
                print("Valid_Accuracy:{:.4f}".format(correct_ratio))

    def one_file_test(self, test_iter, lack, file_num, flag=True):
        """
        测试模型
        :param model: 模型
        :param criterion: 损失函数
        :param test_iter: 测试集迭代器
        :param lack: 待抽样标签
        :param file_num: 文件编号
        :param flag: 是否为最后一个文件
        :return:
        """

        # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("-------one file test----------")
        self.compute_threshold(file_num)

        self.model.eval()
        next_test_index = []

        current_lack_correct = 0
        current_all_correct = 0
        current_lack_len = 0
        current_all_len = 0

        lack_correct_ratio = 0
        ample_correct_ratio = 0

        num_1 = 0
        num_0 = 0

        main_test_file = pd.read_csv("data/my_test.csv", header=None, encoding="utf-8")
        with torch.no_grad():  # 不进行梯度计算
            if flag:
                for index, batch in enumerate(test_iter):
                    
                    target = batch.label.to(DEVICE)
                    data = batch.review.to(DEVICE)

                    if np.array(main_test_file)[index][1] != target[0].item():
                        target = (target - 1) * (-1)
                        # print(np.array(main_test_file)[index][1],batch.label[0].item())
                        # print(index)
                        # print("-------------")
                    if (data.shape[0] < 3):
                        continue
                        
                    outputs = self.model(data)

                    batch_prob = torch.softmax(outputs, dim=1)

                    # -----------------概率三支-------------------------
                    # for prob in batch_prob:
                    #     all_entropy.append(prob[0])
                    #
                    # all_entropy = np.array(all_entropy)
                    # flags1 = np.argwhere((all_entropy >= 0.9) | (all_entropy <= 0.15)).squeeze().tolist()
                    # flags2 = (np.argwhere((0.15 < all_entropy) & (all_entropy < 0.9)).squeeze() + (len(batch) * index)).tolist()
                    # -------------------------------------------------

                    # -------------------三支代价敏感---------------------
                    if ((self.alpha > batch_prob[0][0]) and (batch_prob[0][0] > self.beta)):
                        next_test_index.append(index)
                        if target[0].item() != lack:
                            num_1 += 1
                        else:
                            num_0 += 1
                    else:
                        current_all_len += 1
                        predict_label = batch_prob.argmax(1)[0].item()
                        if ((target[0].item() == lack) and (predict_label == lack)):
                            self.current_lack_correct += 1
                            current_lack_correct += 1

                        if(predict_label == target[0].item()):
                            self.current_all_correct += 1
                            current_all_correct += 1

                        if(target[0].item() == lack):
                            self.current_lack_len += 1
                            current_lack_len += 1
                    # --------------------------------------------------
            else:
                print("-------------final file---------------")
                for index, batch in enumerate(test_iter):
                    target = batch.label.to(DEVICE)
                    data = batch.review.to(DEVICE)
                    if (data.shape[0] < 3):
                        continue
                    if np.array(main_test_file)[index][1] != target[0].item():
                        target = (target - 1) * (-1)
                        # print(np.array(main_test_file)[index][1], batch.label[0].item())
                        # print(index)
                        # print("-------------")

                    # batch.label = (batch.label - 1) * (-1)
                    
                    outputs = self.model(data)
                    prob = torch.softmax(outputs, dim=1)

                    if prob[0][0] < 0.5:
                        predict_label = 1
                    else:
                        predict_label = 0

                    if target[0].item() != lack:
                        num_1 += 1
                    else:
                        num_0 += 1

                    current_all_len += 1
                    if ((target[0].item() == lack) and (predict_label == lack)):
                        self.current_lack_correct += 1
                        current_lack_correct += 1

                    if (predict_label == target[0].item()):
                        self.current_all_correct += 1
                        current_all_correct += 1
                    
                    if (target[0].item() == lack):
                        self.current_lack_len += 1
                        current_lack_len += 1

            if ((current_lack_len) != 0):
                lack_correct_ratio = current_lack_correct / current_lack_len
                print(current_lack_correct, current_lack_len)
            else:
                print("---No Lack Data---")

            if ((current_all_len - current_lack_len) != 0):
                ample_correct_ratio = (current_all_correct - current_lack_correct) / (current_all_len - current_lack_len)
                print((current_all_correct - current_lack_correct), (current_all_len - current_lack_len))
            else:
                print("---No Ample Data---")

            self.TN += current_lack_correct
            self.FP += current_lack_len - current_lack_correct
            self.TP += current_all_correct - current_lack_correct
            self.FN += current_all_len - current_lack_len - (current_all_correct - current_lack_correct)

            print("next 0 and 1:{} {}".format(num_0 ,num_1))
            print("now 0 and 1:{} {}".format(current_lack_len, current_all_len - current_lack_len))

            # test_loss /= len(test_iter)
            # print("Test_Loss:{}".format(test_loss))
            print("Lack_Acc:{:.4f}".format(lack_correct_ratio))
            print("Ample_Acc:{:.4f}\n".format(ample_correct_ratio))

            return next_test_index

    def compute_F1(self):
        Precision = self.TP / (self.TP + self.FP)
        Recall = self.TP / (self.TP + self.FN)
        F1 =  2 * (Precision*Recall) / (Precision + Recall)
        print("F1:{:.4f}".format(F1))

    def main(self, num_time):
        lack_label = find_lack_label(self.all_train_filename)
        # --------------------------------------------------------------------------------

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        LABEL = data.LabelField()
        REVIEW = data.Field(lower=True)

        # 设置表头
        fields = [('review', REVIEW), ('label', LABEL)]

        criterion = torch.nn.CrossEntropyLoss()

        for root, dirs, files in os.walk(self.trainFolder, topdown=False):
            train_files = files

        self.train_file_all_num = len(train_files)

        for k in range(num_time):
            start_time = time.time()
            print("----------第{}次----------".format(k))

            self.model = TextCNN(hidden_size=100, embedding_dim=256, vocab_size=20002, out_size=2)
            self.model=self.model.to(DEVICE)
            self.TN = 0
            self.FP = 0
            self.TP = 0
            self.FN = 0
            self.current_lack_correct = 0
            self.current_all_correct = 0
            self.current_lack_len = 0

            index = []
            flag = True
            file_num = 0
            for trainFile in train_files:
                print("file name: {}".format(trainFile))
                file_num += 1
                if trainFile == train_files[-1]:
                    flag = False

                # self.split_valid(self.trainFolder + "/" + trainFile)
                # extractByFile_fisvdd(self.trainFolder + "/main_train/" + trainFile, self.sampleFolder + "/" + trainFile, True)
                # extractByFile_random(self.trainFolder + "/main_train/" + trainFile, self.sampleFolder + "/" + trainFile, False)

                train_data = data.TabularDataset(
                    path=self.sampleFolder + "/" + trainFile,
                    format='CSV',
                    fields=fields,
                    skip_header=False
                )

                valid_data = data.TabularDataset(
                    path=self.trainFolder + "/valid_files/" + trainFile,
                    format='CSV',
                    fields=fields,
                    skip_header=False
                )

                REVIEW.build_vocab(train_data, max_size=20000)  # 取出频率最大的前20000个单词
                LABEL.build_vocab(train_data)

                if (index != None and len(index) != 0):
                    main_test_file = pd.read_csv("data/my_test.csv", header=None, encoding="utf-8")
                    main_test_file = np.array(main_test_file)[np.array(index)]
                elif file_num == 1:
                    one_test_file = pd.read_csv(self.all_test_filename, header=None, encoding="utf-8")
                    main_test_file = np.array(one_test_file)
                else:
                    break
                
                dataframe = pd.DataFrame({'a_name': main_test_file[:, 0], 'b_name': main_test_file[:, 1]})
                dataframe.to_csv("data/my_test.csv", header=False, encoding="utf-8", sep=",", index=False)
                file = pd.read_csv("data/my_test.csv", header=None, encoding="utf-8")
                
                print("my_test\n", file[1].value_counts())

                temp_test_data = data.TabularDataset(
                    path="data/my_test.csv",
                    format='CSV',
                    fields=fields,
                    skip_header=False
                )

                # 文本批处理
                train_iter, valid_iter = data.BucketIterator.splits((train_data, valid_data),
                                                                    batch_size=64,
                                                                    shuffle=True,
                                                                    sort_within_batch=True,
                                                                    device=DEVICE,
                                                                    sort_key=lambda x: len(x.review))

                test_iter, test_iter1 = data.BucketIterator.splits((temp_test_data, temp_test_data),
                                                                    repeat=False,
                                                                    batch_size=1,
                                                                    device=DEVICE,
                                                                    shuffle=False)
                # for index, batch in enumerate(train_iter):
                #     for j in batch.review:
                #         for i in j:
                #             print(REVIEW.vocab.itos[i])
                #         break
                #     print(batch.label)
                #     break
                # print("------------------")

                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
                self.one_file_train(optimizer, criterion, train_iter, valid_iter, 5)

                index = self.one_file_test(test_iter, lack_label, file_num, flag)

            lack_arr = self.TN / (self.TN + self.FP)
            ample_arr = self.TP / (self.TP + self.FN)
            my_correct_arr = (self.TN + self.TP) / (self.TN + self.FP + self.TP + self.FN)

            print("--------------all files test----------------")
            print(self.TP, self.FP, self.TN, self.FN)
            print(self.current_lack_len, self.current_all_correct, self.current_lack_correct)
            print("Lack_Accuracy:{:.4f}".format(lack_arr))
            print("Ample_Accuracy:{:.4f}".format(ample_arr))
            print("Accuracy:{:.4f}".format(my_correct_arr))
            TC = self.lamb_PN*self.FP + self.lamb_NP*self.FN
            print("AC:{:.4f}".format(TC/(self.TN + self.FP + self.TP + self.FN)))
            self.compute_F1()

            end_time = time.time()
            d_time = end_time - start_time

            print("程序运行时间：%.8s s" % d_time)  # 显示到微秒

import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
            
if __name__ == '__main__':
    sampleFolder = "data/sample_files/Amazon"
    trainFolder = "data/initial_dataset/Amazon/train_files"
    validFolder = "data/initial_dataset/Amazon/train_files/valid_files"
    all_train_filename="data/initial_dataset/Amazon/Amazon_Train.csv"
    all_test_filename="data/initial_dataset/Amazon/Amazon_Test1.csv"
    sys.stdout = Logger(stream=sys.stdout)

    twssc = TWSSC(trainFolder=trainFolder, validFolder=validFolder, sampleFolder=sampleFolder, all_train_filename=all_train_filename, all_test_filename=all_test_filename)
    twssc.main(5)