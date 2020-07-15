import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

# Class to capture all parameters of the problem, setup beta accordingly,
class Chap4():
    
    def __init__(self, n, p, s): #captures all parameters of the problem
        self.n = n
        self.p = p
        self.s = s
        self.beta = np.zeros((p,1))
        for i in range(5):
            self.beta[i] = s-i
        
        self.ev = np.zeros((p))
        self.cov = np.ones((p,p))*0.25
        for i in range(p):
            self.cov[i,i] = 1
    
    def calc_params(self): # function to calculate a sample with size n
        X = np.random.multivariate_normal(self.ev,self.cov,self.n)
        Epsilon = np.random.normal(size = self.n)
        Y = np.dot(X, self.beta)[:,0] + Epsilon
        return Y,X


def dist_cust(beta1, beta2): # calculatest the custom distance
    ret_val = []
    for i in [0,4,49]:
        ret_val.append((beta1[i]-beta2[i])**2)
    return ret_val

# function that calculates the mean of the distances and the number of zeros
def solveChap4(R,model):
    sample = [] # var for collecting the distance of the samples
    zeros = [0,0,0] # var for counting number of zeros in coefficients 0,4,49
    for i in range(R):
        Y, X = model.calc_params()
        lin_model.fit(X,Y)
        lasso_model.fit(X,Y)
        sample.append(dist_cust(lin_model.coef_, lasso_model.coef_))
        
        # count number of zero in coefficients
        if lasso_model.coef_[0] == 0:
            zeros[0] += 1
        if lasso_model.coef_[4] == 0:
            zeros[1] += 1
        if lasso_model.coef_[49] == 0:
            zeros[2] +=1
        
    sample = np.array(sample)
    mse_custom = sample.mean(axis = 0) # this is the sample mean (euclidean distance)
    print("The mse of index 1,5,50 is", mse_custom[0], mse_custom[1], mse_custom[2],",respectively")
    print("The coefficient of index 1, 5, 50 is set to zero", zeros[0],zeros[1],zeros[2]," times, respectively")


if __name__ == "__main__": # can be used as a library without modification
    # Theese are the models used
    lin_model = LinearRegression()
    lasso_model = Lasso()
    
    #Teilaufgabe 4.3
    n = 100
    p = 50
    s = 5
    
    R = 1 # funktioniert ohne Änderung der oberen Definitionen
    model4_1 = Chap4(n,p,s)
    print("For R = 1:")
    solveChap4(R,model4_1)
    
    # Teilaufgabe 4.4
    n = 100
    p = 50
    s = 5
    R = 1000
    model4_1 = Chap4(n,p,s)
    solveChap4(R,model4_1)
    
    # Teilaufgabe 4.5
    R = 1000
    for i in [70,90,110]:
        print(f"With p equal to {i}:")
        model4_5 = Chap4(n,i,s)
        solveChap4(R,model4_5)
    
    # Teilaufgabe 4.6: visualisation
    
    
    
