#coding=utf-8
import numpy as np

class testclass:

    def acc(self,x_lable, y_lable):
        b = []
        for i in range(len(x_lable)):
            if x_lable[i] == y_lable[i]:
                b.append(1)
        train_acc = len(b)
        return train_acc

    def len(self,X, batch_size):
        if(X%batch_size == 0):
            len = X
        else:
            len = (X - (X % batch_size))

        return len