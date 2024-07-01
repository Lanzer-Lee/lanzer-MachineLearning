import numpy as np
from matplotlib import pyplot as plt

class Clustering():
    def __init__(self) -> None:
        self.num_examples=0
        self.data=None
        self.cluster=None
        self.reference=None
        self.num_cluster=0
        self.SS=0
        self.SD=0
        self.DS=0
        self.DD=0
        self.JC=0
        self.FMI=0
        self.RI=0
        self.DBI=0
        self.DI=0

    def minkowski_distance(self,x1,x2,p=2):
        return np.sum(np.abs(x1-x2)**p)**(1/p)
    
    def external_index(self):
        self.SS=0
        self.SD=0
        self.DS=0
        self.DD=0
        for i in range(self.num_examples):
            for j in range(i+1,self.num_examples):
                if self.cluster[i]==self.cluster[j] and self.reference[i]==self.reference[j]:
                    self.SS+=1
                elif self.cluster[i]==self.cluster[j] and self.reference[i]!=self.reference[j]:
                    self.SD+=1
                elif self.cluster[i]!=self.cluster[j] and self.reference[i]==self.reference[j]:
                    self.DS+=1
                elif self.cluster[i]!=self.cluster[j] and self.reference[i]!=self.reference[j]:
                    self.DD+=1
        self.JC=self.SS/(self.SS+self.SD+self.DS)
        self.FMI=np.sqrt(self.SS*self.SS/(self.SS+self.SD)/(self.SS+self.DS))
        self.RI=2*(self.SS+self.DD)/self.num_examples/(self.num_examples-1)
        print('SS,SD,DS,DD=',self.SS,self.SD,self.DS,self.DD)
        print('Jaccard Coefficient=',self.JC)
        print('Fowlkes and Mallows Index=',self.FMI)
        print('Rand Index',self.RI)

    def arange(self,C):
        avg=0
        for i in range(len(C)):
            for j in range(i+1,len(C)):
                avg+=self.minkowski_distance(C[i,:],C[j,:])
        avg=avg*2/len(C)/(len(C)-1)
        return avg
    
    def diam(self,C):
        max_dis=0
        for i in range(len(C)):
            for j in range(i+1,len(C)):
                if self.minkowski_distance(C[i,:],C[j,:])>max_dis:
                    max_dis=self.minkowski_distance(C[i,:],C[j,:])
        return max_dis
    
    def dmin(self,Ci,Cj):
        min_dis=self.minkowski_distance(Ci[0,:],Cj[0,:])
        for i in range(len(Ci)):
            for j in range(len(Cj)):
                if self.minkowski_distance(Ci[i,:],Cj[j,:])<min_dis:
                    min_dis=self.minkowski_distance(Ci[i,:],Cj[j,:])
        return min_dis
    
    def dmean(self,C):
        return np.sum(C,axis=0)/len(C)
    
    def dcen(self,Ci,Cj):
        return self.minkowski_distance(self.dmean(Ci),self.dmean(Cj))

    def internal_index(self):
        dbi=0
        for i in range(self.num_cluster):
            delta_dbi=0
            for j in range(self.num_cluster):
                if i!=j:
                    index_i=np.argwhere(self.cluster==i)
                    index_j=np.argwhere(self.cluster==j)
                    Ci=self.data[index_i.T[0],:]
                    Cj=self.data[index_j.T[0],:]
                    if len(Ci)==0 or len(Cj)==0:
                        continue
                    if (self.arange(Ci)+self.arange(Cj))/self.dcen(Ci,Cj)>delta_dbi:
                        delta_dbi=(self.arange(Ci)+self.arange(Cj))/self.dcen(Ci,Cj)
            dbi+=delta_dbi
        self.DBI=dbi/self.num_cluster
        print('Davies-Bouldin Index=',self.DBI)

class Kmeans(Clustering):
    def __init__(self) -> None:
        super().__init__()
        self.losses=[]
        self.centroids=None

    def random_init(self):
        index=np.random.choice(np.arange(self.num_examples),size=self.num_cluster,replace=False)
        print('random init central:',index)
        self.centroids=self.data[index]
        self.cluster=np.zeros(self.num_examples,dtype=int)
    
    def train(self,max_iter=500):
        update=True
        iter=0
        self.losses=[]
        while update:
            iter+=1
            if iter>max_iter:
                break
            update=False
            C:list[list]=[]
            for i in range(self.num_cluster):
                C.append([])
            for j in range(self.num_examples):
                d=[]
                for i in range(self.num_cluster):
                    d.append(self.minkowski_distance(self.data[j,:],self.centroids[i,:]))
                    lambda_j=np.argmin(d)
                    C[lambda_j].append(self.data[j,:])
                    self.cluster[j]=lambda_j
            for i in range(self.num_cluster):
                new_central=self.dmean(C[i])
                if (new_central!=self.centroids[i]).any():
                    self.centroids[i]=new_central
                    update=True
            loss=0
            for j in range(self.num_examples):
                loss+=self.minkowski_distance(self.data[j,:],self.centroids[self.cluster[j],:])
            self.losses.append(loss)
        return np.array(self.losses)
    
class LVQ(Clustering):
    def __init__(self,K=4) -> None:
        super().__init__()
        self.num_cluster=K
        self.labels=None
        self.tlabels=None
        self.centroids=None

    def random_init(self):
        q=[]
        for i in range(self.num_cluster):
            index=np.argwhere(self.labels==i).T[0]
            q.append(np.random.choice(index,size=1,replace=False)[0])
        self.tlabels=self.labels[q]
        self.centroids=self.data[q]
        self.cluster=self.labels
        #index=np.random.choice(np.arange(self.num_examples),size=self.num_cluster,replace=False)
        #print('random init central:',index)
        #self.centroids=self.data[index]
        #self.cluster=np.zeros(self.num_examples,dtype=int)

    def train(self,max_iter=400,lr=0.1):
        iter=0
        while iter<max_iter:
            j=np.random.choice(np.arange(self.num_examples),size=1,replace=False)[0]
            d=[]
            for i in range(self.num_cluster):
                d.append(self.minkowski_distance(self.data[j,:],self.centroids[i,:]))
            i=np.argmin(d)
            if self.labels[j]==self.tlabels[i]:
                self.centroids[i,:]+=lr*(self.data[j,:]-self.centroids[i,:])
            else:
                self.centroids[i,:]-=lr*(self.data[j,:]-self.centroids[i,:])
                iter+=1
        for j in range(self.num_examples):
            d=[]
            for i in range(self.num_cluster):
                d.append(self.minkowski_distance(self.data[j,:],self.centroids[i,:]))
            i=np.argmin(d)
            self.cluster[j]=i

class KmeansPoint(Kmeans):
    def __init__(self,K=4) -> None:
        super().__init__()
        self.num_cluster=K
        self.load_data()
        self.random_init()

    def load_data(self):
        dataset=np.loadtxt('Clustering/kmeans_data.csv', delimiter=',')
        self.num_examples=len(dataset)
        self.data=dataset
        print('num examples=',self.num_examples) 

    def view_data(self):
        plt.figure()
        colors=np.array(['#0072BD','#D95319','#EDB120','#77AC30'])
        plt.scatter(self.data[:,0],self.data[:,1],color=colors[self.cluster])
        if self.centroids is not None:
            plt.scatter(self.centroids[:,0],self.centroids[:,1],color=colors[:len(self.centroids)],marker='+',s=150)
        plt.show()

    def view_loss(self):
        plt.figure()
        plt.plot(np.arange(len(self.losses)),self.losses)
        plt.show()

class LVQPoint(LVQ):
    def __init__(self) -> None:
        super().__init__()
        self.load_data()
        self.random_init()

    def load_data(self):
        dataset=np.loadtxt('Clustering/kmeans_data.csv', delimiter=',')
        self.num_examples=len(dataset)
        self.data=dataset
        kmeans=KmeansPoint()
        kmeans.train()
        self.labels=kmeans.cluster
        print('num examples=',self.num_examples) 


    def view_data(self):
        plt.figure()
        colors=np.array(['#0072BD','#D95319','#EDB120','#77AC30'])
        plt.scatter(self.data[:,0],self.data[:,1],color=colors[self.cluster])
        if self.centroids is not None:
            plt.scatter(self.centroids[:,0],self.centroids[:,1],color=colors[:len(self.centroids)],marker='+',s=150)
        plt.show()

def main():
    kmeans=KmeansPoint()  
    kmeans.train()
    kmeans.internal_index()

    lvq=LVQPoint()
    lvq.view_data()
    lvq.train()
    lvq.view_data()
    lvq.internal_index()

if __name__=='__main__':
    main()