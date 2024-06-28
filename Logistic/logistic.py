import numpy as np
from matplotlib import pyplot as plt
from evaluate import Evaluate

class Logistic():
    def __init__(self) -> None:
        self.features=None
        self.labels=None
        self.num_examples=0
        self.num_features=0
        self.features_train=None
        self.features_test=None
        self.labels_train=None
        self.labels_test=None
        self.theta=None

    def logistic(self,z):
        return 1/(1+np.exp(-z))
    
    def train(self,num_steps,lr,l2_coef):
        evaluate=Evaluate()
        self.theta=np.random.normal(size=[self.num_features])
        for i in range(num_steps):
            pred=self.logistic(self.features_train @ self.theta)
            grad=-self.features.T @ (self.labels_train-pred)+l2_coef*self.theta
        

class LogisticPoint(Logistic):
    def __init__(self) -> None:
        super().__init__()
        self.load_data()

    def load_data(self):
        lines=np.loadtxt('Logistic/lr_dataset.csv',delimiter=',',dtype=float)
        self.features=lines[:,:2]
        self.labels=lines[:,2]
        self.num_examples,self.num_features=self.features.shape

    def split_data(self,ratio=0.7):
        np.random.seed(0)
        index=np.random.permutation(self.num_examples)
        split=int(self.num_examples*ratio)
        self.features=self.features[index]
        self.labels=self.labels[index]
        self.features_train=self.features[:split]
        self.labels_train=self.labels[:split]
        self.features_test=self.features[split:]
        self.labels_test=self.labels[split:]


    def view_data(self):
        index_pos=np.where(self.labels==1)
        index_neg=np.where(self.labels==0)
        plt.figure()
        plt.scatter(self.features[index_pos,0],self.features[index_pos,1],marker='o',color='coral',s=10)
        plt.scatter(self.features[index_neg,0],self.features[index_neg,1],marker='x',color='blue',s=10)
        plt.xlabel('x1 axis')
        plt.ylabel('x2 axis')
        plt.show()

def main():
    logi=LogisticPoint()
    #logi.view_data()
    logi.train(0,0,0)
    print(logi.theta)

if __name__=='__main__':
    main()