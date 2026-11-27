import numpy as np

def forward(x,a,b):
    y = a*x + b
    return y



def loss(yp,ya):
    return (yp-ya)

def update(loss,a,b,n,x):
    # dloss/da = 2*(yp-ya)*x
    # dloss/db = 2*(yp-ya)*1
    grada = np.mean(2*(loss)*x)
    gradb = np.mean(2*(loss)*1)
    a = a - grada*n
    b = b - gradb*n
    return a,b


X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],dtype=float)

Y = np.array([25, 30, 38, 42, 51, 57, 63, 70, 76, 85],dtype=float)

graph_mean = []

a,b=0.0,0.0

N = 100

for epoch in range(1,N):
    yp = forward(X,a,b)
    lossc = loss(yp,Y)
    graph_mean.append(np.average(lossc**2))
    a,b = update(lossc,a,b,np.array([0.01]),X)


import matplotlib.pyplot as plt

plt.plot([i for i in range(1,N)],graph_mean)
plt.savefig("linreg")






      

