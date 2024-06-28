import numpy as np
from matplotlib import pyplot as plt

class Evaluate():
    def __init__(self,labels_true,labels_pred) -> None:
        self.labels_true=np.array(labels_true)
        self.labels_pred=np.array(labels_pred)
        self.train_losses=[]
        self.test_losses=[]
        self.train_acc=[]
        self.test_acc=[]
        self.train_auc=[]
        self.test_auc=[]
        self.TP=0
        self.TN=0
        self.FP=0
        self.FN=0
        self.TPR=0
        self.FPR=0
        self.ACC=0
        self.REC=0
        self.PREC=0
        self.F1=0
        self.AUC=0

    def acc(self,train=True):
        if train==True:
            self.train_acc.append(np.mean(self.labels_true==self.labels_pred))
        else:
            self.test_acc.append(np.mean(self.labels_true==self.labels_pred))
    
    def confusion(self):
        self.TP=np.sum((self.labels_true==self.labels_pred)*(self.labels_true>0))
        self.TN=np.sum(((self.labels_true==self.labels_pred)*(self.labels_true<=0)))
        self.FP=np.sum(((self.labels_true!=self.labels_pred)*(self.labels_true<=0)))
        self.FN=np.sum(((self.labels_true!=self.labels_pred)*(self.labels_true>0)))

    def statistic(self,show=True):
        self.confusion()
        self.ACC=(self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
        self.TPR=self.TP/(self.TP+self.FN)
        self.FPR=self.FP/(self.FP+self.TN)
        self.REC=self.TPR
        self.PREC=self.TP/(self.TP+self.FP)
        self.F1=self.f_beta_score()
        self.AUC=self.auc()
        if(show):
            print('TP=',self.TP)
            print('TN=',self.TN)
            print('FP=',self.FP)
            print('FN=',self.FN)
            print('accuracy=',self.ACC)
            print('TPR=',self.TPR)
            print('FPR=',self.FPR)
            print('Recall rate=',self.REC)
            print('Precision=',self.PREC)
            print('F1 score=',self.F1)
            print('AUC=',self.AUC)

    def f_beta_score(self,beta=1):
        return (1+beta**2)*(self.PREC+self.REC)/(beta**2*self.PREC+self.REC)

    def auc(self):
        index=np.argsort(self.labels_pred)[::-1]
        tp=np.cumsum(self.labels_true[index])
        fp=np.cumsum(1-self.labels_true[index])
        tpr=tp/tp[-1]
        fpr=fp/fp[-1]
        res=0.0
        tpr=np.concatenate([[0],tpr])
        fpr=np.concatenate([[0],fpr])
        for i in range(1,len(fpr)):
            res+=(fpr[i]-fpr[i-1])*tpr[i]
        return res

def main():
    labels_true=[1,1,0,0]
    labels_pred=[1,0,0,1]
    evaluate=Evaluate(labels_true,labels_pred)
    evaluate.statistic()

if __name__=='__main__':
    main()
