from matplotlib import pyplot as plt
import numpy as np

class KNN():
    def __init__(self,k,label_num):
        self.k=k
        self.label_num=label_num
        self.x_train=None
        self.x_test=None
        self.y_train=None
        self.y_test=None

    def predict(self,x):
        dis=list(map(lambda a:self.distance(a,x),self.x_train))
        indices=np.argsort(dis)[:self.k]
        label_preds=self.y_train[indices]
        return np.argmax(np.bincount(label_preds))
    
    def test(self):
        accuracy=0
        for i,feature in enumerate(self.x_test):
            label_pred=self.predict(feature)
            if(label_pred==self.y_test[i]):
                accuracy+=1
        return accuracy/self.y_test.shape[0]

    def distance(self,a,b):
        return np.sqrt(np.sum((a-b)**2))

class KNN_Mnist(KNN):
    def __init__(self,k=5,ratio=0.8):
        super().__init__(k,10)
        self.feature,self.label=self.load_data()
        self.split_data(ratio)

    def load_data(self):
        features=np.loadtxt('mnist_x',delimiter=' ',dtype=np.uint8)
        labels=np.loadtxt('mnist_y',dtype=np.uint8)
        print('mnist_x shape=',features.shape)
        print('mnist_y shape=',labels.shape)
        return features,labels
    
    def split_data(self,ratio):
        split=int(self.feature.shape[0]*ratio)
        index=np.random.permutation(np.arange(self.feature.shape[0]))
        self.feature=self.feature[index]
        self.label=self.label[index]
        self.x_train=self.feature[:split]
        self.x_test=self.feature[split:]
        self.y_train=self.label[:split]
        self.y_test=self.label[split:]
        print('feature train data shape=',self.x_train.shape)
        print('label train data shape=',self.y_train.shape)
        print('feature test data shape=',self.x_test.shape)
        print('label tes data shape=',self.y_test.shape)

    def view_data(self,index):
        if(index<0 or index>=self.lable.shape[0]):
            raise IndexError('Index exceed bound')
        img=np.reshape(np.array(self.feature[index],dtype=int),[28,28])
        plt.figure()
        plt.imshow(img,cmap='gray')
        plt.show()

def main():
    knn=KNN_Mnist()
    print(knn.test())
    

if __name__=='__main__':
    main()
    