import numpy as np
from matplotlib import pyplot as plt

class Linear():
    def __init__(self) -> None:
        self.features_num=0
        self.samples_num=0
        self.data=None
        self.data_train=None
        self.data_test=None
        self.x_train=None
        self.x_test=None
        self.y_train=None
        self.y_test=None
        self.W=0
        self._mean=0
        self._std=0

    def fit(self):
        self._mean=np.zeros([1,self.features_num+1])
        self._std=np.zeros([1,self.features_num+1])
        for i in range(self.features_num+1):
            self._mean[0,i]=np.mean(self.data_train[:,i])
            self._std[0,i]=np.std(self.data_train[:,i])
            self.data_train[:,i]=(self.data_train[:,i]-self._mean[0,i])/self._std[0,i]
            self.data_test[:,i]=(self.data_test[:,i]-self._mean[0,i])/self._std[0,i]
        self.x_train=self.data_train[:,:-1]
        self.y_train=self.data_train[:,-1]
        self.x_test=self.data_test[:,:-1]
        self.y_test=self.data_test[:,-1]
        self.x_train=np.concatenate([self.x_train,np.ones([len(self.x_train),1])],axis=-1)
        self.x_test=np.concatenate([self.x_test,np.ones([len(self.x_test),1])],axis=-1)
        print('data has been normalrized')

    def train(self,num_epoch=20,lr=0.01,batch_size=32,analytical=False):
        if(analytical==True):
            self.W=np.linalg.inv(self.x_train.T @ self.x_train) @ self.x_train.T @ self.y_train
            self.test()
        else:
            opt=Optimazer()
            opt=Optimazer()
            opt.load_data(self.x_train,self.y_train,self.x_test,self.y_test)
            loss=opt.SGD(num_epoch,lr,batch_size)
            print(opt.W)
            opt.view_loss(num_epoch,loss[0],loss[1])

    def test(self):
        predictions=self.x_test @ self.W 
        RMSE=np.sqrt(np.sum((predictions-self.y_test)**2)/self.y_test.shape[0])
        print('RMSE=',RMSE)
        predictions=predictions*self._std[0,-1]+self._mean[0,-1]
        return predictions       

class LinearHouse(Linear):
    def __init__(self,ratio=0.8) -> None:
        super().__init__()
        self.load_data()
        self.split_data(ratio)
        self.fit()
        
    def load_data(self):
        lines=np.loadtxt('Linear/USA_Housing.csv',delimiter=',',dtype=str)
        self.header=lines[0]
        lines=lines[1:].astype(float)
        self.data=lines
        self.samples_num=lines.shape[0]
        self.features_num=lines.shape[1]-1
        print('data features:',','.join(self.header[:-1]))
        print('data labels:',self.header[-1])
        print('examples num:',len(lines))
        print('features num:',self.features_num)

    def split_data(self,ratio):
        split=int(self.samples_num*ratio)
        np.random.seed(0)
        self.data=np.random.permutation(self.data)
        self.data_train=self.data[:split]
        self.data_test=self.data[split:]

    def view_data(self,x,y):
        plt.figure()
        plt.scatter(x,y)
        plt.show()

class Optimazer():
    def __init__(self) -> None:
        self.x_train=None
        self.x_test=None
        self.y_train=None
        self.y_test=None
        self.W=None

    def load_data(self,x_train,y_train,x_test,y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test

    def batch_generator(self,x,y,batch_size,shuffle=True):
        batch_count=0
        if(shuffle):
            index=np.random.permutation(len(x))
            x=x[index]
            y=y[index]
        while(True):
            start=batch_count*batch_size
            end=min(start+batch_size,len(x))
            if(start>=end):
                break
            batch_count+=1
            yield x[start:end],y[start:end]

    def SGD(self,num_epoch,lr,batch_size):
        self.W=np.random.normal(size=self.x_train.shape[1])
        train_losses=[]
        test_losses=[]
        for _ in range(num_epoch):
            batch_data=self.batch_generator(self.x_train,self.y_train,batch_size,shuffle=True)
            train_loss=0
            for batch_x,batch_y in batch_data:
                grad=batch_x.T @ (batch_x @ self.W-batch_y)
                self.W-=lr*grad/len(batch_x)
                train_loss+=np.sum((batch_x @ self.W-batch_y)**2)
            train_loss=np.sqrt(train_loss/len(self.x_train))
            train_losses.append(train_loss)
            test_loss=np.sqrt(np.mean((self.x_test @ self.W - self.y_test)**2))
            test_losses.append(test_loss)
        return train_losses,test_losses
    
    def view_loss(self,num_epoch,train_loss,test_loss=None):
        plt.figure()
        plt.plot(np.arange(num_epoch),train_loss,label='train loss')
        plt.plot(np.arange(num_epoch),test_loss,label='test loss')
        plt.xlabel('epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.show()

def main():
    linear=LinearHouse()
    linear.train(analytical=True)
    linear.train()
    
if __name__=='__main__':
    main()