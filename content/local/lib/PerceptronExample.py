from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import itertools

def sigmoide(u):
	g = np.exp(u)/(1 + np.exp(u))
	return g

def Plot_Perceptron():
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	X2 = X[:100][:,:2]
	y2 = y[:100]
	fig, (ax0, ax1) = plt.subplots(1,2)
	ax0.scatter(X2[:,0], X2[:,1], c=y2, cmap="Accent")

	#Aprendizaje
	MaxIter = 100000
	w = np.ones(3).reshape(3, 1)
	eta = 0.001
	N = len(y2)
	Error =np.zeros(MaxIter)
	Xent = np.concatenate((X2,np.ones((100,1))),axis=1)
	for i in range(MaxIter):
		tem = np.dot(Xent,w)
		tem2 = sigmoide(tem.T)-np.array(y2)
		Error[i] = np.sum(abs(tem2))/N
		tem = np.dot(Xent.T,tem2.T)
		wsig = w - eta*tem/N
		w = wsig
	print("Weights:")
	print(w)
	print('Error=',Error[-1])
	#Grafica de la frontera encontrada
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	X2 = X[:100][:,:2]
	y2 = y[:100]
	ax1.scatter(X2[:,0], X2[:,1], c=y2,cmap="Accent")
	x1 = np.linspace(4,8,20)
	x2 = -(w[0]/w[1])*x1 - (w[2]/w[1])
	ax1.plot(x1,x2,'k')

def cross_entropy(X2,t,w):
    epsilon=1e-12
    Xent = np.concatenate((X2,np.ones((X2.shape[0],1))),axis=1)
    predictions = np.clip(sigmoide(np.dot(Xent,w.T)), epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -(np.sum(t*np.log(predictions+1e-9))+np.sum((1-t)*np.log(1-predictions)))/N
    return ce

def Gradiente_v2(X2,y2,MaxIter = 100000, eta = 0.001, batch = None):
    g = []
    loss_history = []
    w = np.array([20,-10,-90])
    N = len(y2)
    if batch is not None:
        batch_size = batch
    else:
        batch_size = N
    Nbatch = np.floor(N/batch_size)
    y2 = np.array(y2)
    Error =np.zeros(MaxIter)
    Xent = np.concatenate((X2,np.ones((100,1))),axis=1)
    indx = np.random.permutation(N)
    for i in range(MaxIter):
        for j in range(int(Nbatch)):
            if j < Nbatch - 2:
                Xbatch = Xent[indx[j*batch_size:(j+1)*batch_size],:]
                y2_batch = y2[indx[j*batch_size:(j+1)*batch_size]]
            else:
                Xbatch = Xent[indx[j*batch_size:],:]
                y2_batch = y2[indx[j*batch_size:]]
            
            tem = np.dot(Xbatch,w)
            tem2 = sigmoide(tem.T)-y2_batch
            tem = np.dot(Xbatch.T,tem2.T)
            w = w - eta*tem/batch_size
            g.append(w)
            loss_history.append(cross_entropy(X2,y2,w))
    return w, g, loss_history

def n_grad(X2,t,w):
    N = X2.shape[0]
    Xent = np.concatenate((X2,np.ones((X2.shape[0],1))),axis=1)
    return 2*Xent.T.dot(sigmoide(Xent.dot(w))-t)/N

def plot_cost(cost, t0_range, t1_range, vx=None,vy=None, vz=None):
    k0,k1 = 60,60

    t0 = np.linspace(t0_range[0], t0_range[1], k0)
    t1 = np.linspace(t1_range[0], t1_range[1], k1)

    p = np.zeros((k0,k1))

    for i,j in itertools.product(range(k0), range(k1)):
        p[i,j] = np.log(cost(np.r_[t0[i],t1[j], vz]))

    plt.contourf(t0, t1, p.T, cmap=plt.cm.hot, levels=np.linspace(np.min(p), np.max(p), 20))
    plt.ylabel(r"$w_2$")
    plt.xlabel(r"$w_1$")
    plt.title("loss")
    plt.colorbar()

    if vx is not None:
        plt.axvline(vx, color="white")
    if vy is not None:
        plt.axhline(vy, color="white")

def Plot_SGD_trajectory():
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	X2 = X[:100][:,:2]
	y2 = y[:100]
	loss = lambda t: cross_entropy(X2,y2,t)
	eta = 1.2
	plt.figure(figsize=(15,6))
	plt.subplot(131)
	w, g, _ = Gradiente_v2(X2,y2,MaxIter = 10000, eta = eta, batch = 1)
	plot_cost(loss, (10,60), (-5,-70), vx=w[0], vy=w[1], vz=w[2])
	g = np.r_[g]
	plt.plot(g[::100,0], g[::100,1], color="blue")
	plt.scatter(g[::100,0], g[::100,1], color="blue", s=20)
	plt.scatter(g[-1,0], g[-1,1], marker="x", color="white", s=200)
	plt.title('Stochastic Gradient Descent')

	plt.subplot(132)
	w, g, _ = Gradiente_v2(X2,y2,MaxIter = 10000, eta = eta, batch = 5)
	plot_cost(loss, (10,60), (-5,-70), vx=w[0], vy=w[1], vz=w[2])
	g = np.r_[g]
	plt.plot(g[::100,0], g[::100,1], color="blue")
	plt.scatter(g[::100,0], g[::100,1], color="blue", s=20)
	plt.scatter(g[-1,0], g[-1,1], marker="x", color="white", s=200)
	plt.title('Minibatch Gradient Descent')

	plt.subplot(133)
	w, g, _ = Gradiente_v2(X2,y2,MaxIter = 10000, eta = eta, batch = None)
	plot_cost(loss, (10,60), (-5,-70), vx=w[0], vy=w[1], vz=w[2])
	g = np.r_[g]
	plt.plot(g[::100,0], g[::100,1], color="blue")
	plt.scatter(g[::100,0], g[::100,1], color="blue", s=20)
	plt.scatter(g[-1,0], g[-1,1], marker="x", color="white", s=200)
	plt.title('Batch Gradient Descent')
