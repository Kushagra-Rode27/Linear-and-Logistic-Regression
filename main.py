import argparse
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest,f_classif,f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error,mean_absolute_error
def mse(y_pred, y_true,reg_term):
    return np.mean((y_pred - y_true)**2) + reg_term

def mae(y_pred, y_true,reg_term):
    return np.mean(np.abs(y_pred - y_true)) + reg_term

def output_scores(sample,scores,file) : 
    dataf = {"sample" : sample,"score" : scores.flatten()}
    df = pd.DataFrame(dataf)
    if(os.path.isdir(file)) : 
        df.to_csv(os.path.join(file,"out.csv"),header=False, index=False)
    else : 
         df.to_csv(file, header=False, index=False)
#section 3.1
def gradient_descent(X, y, lr, maxit,thresh,X_val,y_val) : 
    
    theta = np.zeros((len(X[0])+1, 1))
    X_new = np.c_[np.ones((len(X), 1)), X]
    X_val_new = np.c_[np.ones((len(X_val), 1)), X_val]
    y_new = np.reshape(y,(len(y),1))
    y_val_new = np.reshape(y_val,(len(y_val),1))
    m = len(X)
    cost_lst_train = []
    cost_lst_val = []
    if(thresh == 0):
        for i in range(maxit):
            y_pred = X_new.dot(theta)
            y_pred_val = X_val_new.dot(theta)
            cost = np.mean((y_pred - y_new)**2) 
            cost_val = np.mean((y_pred_val - y_val_new)**2) 
            gradients = 2* (X_new.T.dot(y_pred - y_new) / m)
            theta = theta - lr * gradients
            cost_lst_train.append(cost)
            cost_lst_val.append(cost_val)
    else : 
        count = 0
        for i in range(maxit):
            count += 1
            y_pred = X_new.dot(theta)
            y_pred_val = X_val_new.dot(theta)
            cost = np.mean((y_pred - y_new)**2) 
            cost_val = np.mean((y_pred_val - y_val_new)**2) 
            gradients = 2* (X_new.T.dot(y_pred - y_new) / m)
            theta = theta - lr * gradients
            cost_lst_train.append(cost)
            cost_lst_val.append(cost_val)
                
            if(i > 0) : 
                reltol = abs(cost_lst_val[-1] - cost_lst_val[-2]) 
                reltol /= cost_lst_val[-2]
                if(reltol < thresh):
                    break
    return theta,cost_lst_train,cost_lst_val
#section 3.2
def ridge_reg(X,y,lr,maxit,lam,thresh,X_val,y_val) : 
    theta = np.zeros((len(X[0]), 1))
    y_new = np.reshape(y,(len(y),1))
    y_val_new = np.reshape(y_val,(len(y_val),1))
    m = len(X)
    cost_lst_train = []
    cost_lst_val = []
    mse_lst_train = []
    mse_lst_val = [] 
    if(thresh == 0):
        for i in range(maxit):
            y_pred = X.dot(theta)
            y_pred_val = X_val.dot(theta)
            reg_term = lam * np.sum(theta**2)
            cost = np.mean((y_pred - y_new)**2) 
            cost_val = np.mean((y_pred_val - y_val_new)**2)
            gradients = 2 * (X.T.dot(y_pred - y_new)) / m + (2 * lam * theta)
            theta = theta - lr * gradients
            cost_lst_train.append(cost + reg_term)
            cost_lst_val.append(cost_val + reg_term)
            mse_lst_train.append(cost)
            mse_lst_val.append(cost_val)
    else:
        count = 0
        for i in range(maxit):
            count += 1
            y_pred = X.dot(theta)
            y_pred_val = X_val.dot(theta)
            reg_term = lam * np.sum(theta**2)
            cost = np.mean((y_pred - y_new)**2)
            cost_val = np.mean((y_pred_val - y_val_new)**2)
            gradients =  2 * (X.T.dot(y_pred - y_new)) / m + (2 * lam * theta)
            theta = theta - lr * gradients
            cost_lst_train.append(cost + reg_term)
            cost_lst_val.append(cost_val +  reg_term)
            
            mse_lst_train.append(cost)
            mse_lst_val.append(cost_val)
            if(i > 0) : 
                reltol = abs(cost_lst_val[-1] - cost_lst_val[-2]) 
                reltol /= cost_lst_val[-2]
                if(reltol < thresh):
                    break    
    return theta,mse_lst_train,mse_lst_val
#section 3.3
def scikit_lib(X,y,ridge,lam) :
    if(ridge) : 
        model = Ridge(alpha=lam)
        model.fit(X, y) 
        return model
    theta = LinearRegression().fit(X,y)
    return theta
#section 3.4
def sel_feature(X,y) :
    
    X_new = SelectKBest(k=1000).fit_transform(X, y)
    return X_new

def sel_model(X,y) : 
    ridge = Ridge(alpha=5.0)
    selector = SelectFromModel(ridge, threshold=-np.inf, max_features=10)
    X_new = selector.fit_transform(X, y)
    return X_new

#section 3.5

def class_hypothesis(curr_theta, X, all_theta): 
    numer = np.exp(X.dot(all_theta[curr_theta].T))
    deno = np.sum(np.exp(X.dot(all_theta.T)), axis=1)
    probs = (numer) / (1 + deno)
    return probs

def cost_fn(curr_theta, X, all_theta, y):
    y1 = class_hypothesis(curr_theta, X, all_theta)
    y1 = np.array(y1,dtype=np.float64)
    y_new = np.where(y == curr_theta + 1, 1, 0)
    cost = - np.sum(y_new * np.log(y1) + (1 - y_new) * np.log(1 - y1))
    return cost

def gradient(curr_theta, X, all_theta, y): 
    y1 = class_hypothesis(curr_theta, X, all_theta)
    y_new = np.where(y == curr_theta + 1, 1, 0)
    return X.T.dot(y1 - y_new)

def log_regression(X, y, lr, maxit,X_val,y_val):
    X_new = np.c_[np.ones((len(X), 1)), X]
    all_theta = np.zeros((8, X.shape[1] + 1))
    cost_lst_train = []
    cost_lst_val = []
    # loss_alltheta = [[] for _ in range(8)]
    for i in range(maxit):
        temp = all_theta.copy()
        
        for j in range(8): 
            gradients = gradient(j, X_new, all_theta, y)
            temp[j] = all_theta[j] - lr * gradients  
            
        all_theta = temp
        # for j in range(8) :
        #     cost = cost_fn(j,X_new,all_theta,y)
        #     loss_alltheta[j].append(cost)

        #for mse error
        curr_class_train = prediction(X,all_theta)
        cost_train = mse(curr_class_train,y,0)
        cost_lst_train.append(cost_train)
        curr_class_val = prediction(X_val,all_theta)
        cost_val = mse(curr_class_val,y_val,0)
        cost_lst_val.append(cost_val)

    return all_theta,cost_lst_train, cost_lst_val

def prediction(X, all_theta):
    X_new = np.c_[np.ones((len(X), 1)), X]
    h_curr = np.exp(X_new.dot(all_theta.T))
    sum_prob = np.sum(h_curr, axis=1)
    max_prob = np.max(h_curr, axis=1)
    pred_class = np.argmax(h_curr, axis=1) + 1
    pred_class[sum_prob < 1 - max_prob] = 9
    return pred_class



#section 3.6
def visualisation(cost_train,cost_val,section) :
    iters = len(cost_train)
    # print(cost_train)
    # print(cost_val)
    # plt.semilogy()
    plt.plot(np.arange(1 ,iters),cost_train[1:],label='train', color = 'red')
    # plt.plot(np.arange(1,iters),cost_val[1:],label='validation', color = 'green')
    plt.title('MSE Graph')
    # plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('MSE')
    plt.savefig(f'{section}_visu.png')



#main calling function
if(__name__ == "__main__") :
    parser = argparse.ArgumentParser()
    parser.add_argument("-trp", "--train_path", help="path of the training file")
    parser.add_argument("-vp", "--val_path", help="path of the validation file")
    parser.add_argument("-tsp", "--test_path", help="path of the test file")
    parser.add_argument("-op", "--out_path", help="path of the output file")
    parser.add_argument("-s", "--section", help="section of method")

    args = parser.parse_args()

    tf = pd.read_csv(args.train_path, header=None)
    vf = pd.read_csv(args.val_path, header=None)
    tstf = pd.read_csv(args.test_path, header=None)

    if(args.section != None) :
        args.section = int(args.section)
    X = tf.values[:,2:]
    y = tf.values[:,1]
    sample = tf.values[:,0]

    X_val = vf.values[:,2:]
    y_val = vf.values[:,1]
    sample_val = vf.values[:,0]

    X_test =  tstf.values[:,1:]
    sample_test = tstf.values[:,0]
    
    if(args.section == 1):
        normalise = False
        split = False
        half = False
        if(normalise) : 
            X = (X - np.mean(np.array(X).astype(np.float64), axis=0)) / np.std(np.array(X).astype(np.float64), axis=0)
            X_val = (X_val - np.mean(np.array(X_val).astype(np.float64), axis=0)) / np.std(np.array(X_val).astype(np.float64), axis=0)  
        
        if(split) : 
            n = len(X)
            k = 3*(n//4)
            X = X[:k,:]
            y = y[:k]
            
        if(half) :
           n = len(X) 
           k = n//2
           X1 = X[:k,:]
           y1 = y[:k]
           X2 = X[k:,:]
           y2 = y[k:]
        #calling function
        lr = 0.001
        iters = 10000
        thresh = 0.0001
        theta,cost_train,cost_val = gradient_descent(X,y,lr,iters,thresh,X_val,y_val)
        output_scores(sample_test,np.c_[np.ones((len(X_test), 1)), X_test].dot(theta),args.out_path)
        # y_train = np.reshape(y,(len(y),1))
        # y_train_pred = np.c_[np.ones((len(X), 1)), X].dot(theta)
        # y_test_pred = np.c_[np.ones((len(X_test), 1)), X_test].dot(theta)
        # cost_tot = []
        # cost_avg = []
        # for i in range(1,10) : 
        #     err = 0
        #     cnt = 0
        #     for j in range(len(y_train)) : 
        #         if(y_train[j][0] == i) : 
        #             cnt += 1
        #             err += ((i - y_train_pred[j][0])**2)
        #     cost_tot.append(err)
        #     if(cnt != 0):
        #         err/=cnt
        #     cost_avg.append(err)
        
        # plt.plot(np.arange(1 ,10),cost_avg,label='Avg. MSE')
        # plt.plot(np.arange(1 ,10),cost_tot,label='Total MSE')
        
        # plt.title('MSE vs score')
        # plt.legend()
        # plt.xlabel('Score')
        # plt.ylabel('MSE loss')
        # plt.savefig(f'6_6_1t_visu.png')

        # train_mse = mse(y_train_pred, y_train,0)
        # train_mae = mae(y_train_pred, y_train,0)
        # print(train_mse)
        # print(train_mae)

        # # y_val_o = np.reshape(y_val,(len(y_val),1))
        # y_val_pred = X_val.dot(theta)

        # # val_mse = mse(y_val_pred, y_val_o,0)
        # # val_mae = mae(y_val_pred, y_val_o,0)
        # # print(val_mse)
        # # print(val_mae)

        # # visualisation(cost_train,cost_val,"6_3_0.75_rt0.00001")

        # # theta2,cost_train2,cost_val2 = gradient_descent(X2,y2,lr,iters,thresh,X_val,y_val)
        # theta2,cost_train2,cost_val2 = ridge_reg(X2,y2,lr,iters,25,thresh,X_val,y_val)


        # # y_train = np.reshape(y2,(len(y2),1))
        # y_train_pred2 = X.dot(theta2)
        

        # # train_mse = mse(y_train_pred2, y_train,0)
        # # train_mae = mae(y_train_pred2, y_train,0)
        # # print(train_mse)
        # # print(train_mae)

        # # y_val_o = np.reshape(y_val,(len(y_val),1))
        # # y_val_pred2 = np.c_[np.ones((len(X_val), 1)), X_val].dot(theta2)
        # y_val_pred2 = X_val.dot(theta2)

        # # val_mse = mse(y_val_pred2, y_val_o,0)
        # # val_mae = mae(y_val_pred2, y_val_o,0)
        # # print(val_mse)
        # # print(val_mae)


        # # visualisation(cost_train2,cost_val2,"6_4_2")

        # print(mae(y_train_pred,y_train_pred2,0))
        # print(mae(y_val_pred,y_val_pred2,0))
        
    elif(args.section == 2) :
        lr = 0.001
        iters = 10000
        lam = 5
        thresh = 0.00001
        theta,cost_train,cost_val = ridge_reg(X,y,lr,iters,lam,thresh,X_val,y_val)
        output_scores(sample_test,X_test.dot(theta),args.out_path)
        # reg_term = lam * np.sum(theta**2)

        # y_train = np.reshape(y,(len(y),1))
        # y_train_pred = X.dot(theta)
        # train_mse = mse(y_train_pred, y_train,0)
        # train_mae = mae(y_train_pred, y_train,0)
        # print(train_mse)
        # print(train_mae)

        # y_val_o = np.reshape(y_val,(len(y_val),1))
        # y_val_pred = X_val.dot(theta)

        # val_mse = mse(y_val_pred, y_val_o,0)
        # val_mae = mae(y_val_pred, y_val_o,0)
        # print(val_mse)
        # print(val_mae)

        # visualisation(cost_train,cost_val,"3.2_25_rt0.00001")
        
    elif(args.section == 5) :
       lr = 0.001
       iters = 500
       X = np.array(X,dtype=np.float64)
       X_val = np.array(X_val,dtype=np.float64)
       X_test = np.array(X_test,dtype=np.float64)
       all_theta,cost_train,cost_val = log_regression(X,y,lr,iters,X_val,y_val)
       out = prediction(X_test,all_theta)
       output_scores(sample_test,out,args.out_path)
    #    visualisation(cost_train,cost_val,5)

    #    out = prediction(X,all_theta)
    #    out_val = prediction(X_val,all_theta)

    #    train_mse = mse(out, y,0)
    #    train_mae = mae(out, y,0)
    #    print(train_mse)
    #    print(train_mae)

    #    val_mse = mse(out_val, y_val,0)
    #    val_mae = mae(out_val, y_val,0)
    #    print(val_mse)
    #    print(val_mae)

    #    for j in range(0,8) : 
    #     plt.plot(np.arange(1 ,iters),loss_alltheta[j][1:],label=f'theta{j+1}')

    #    plt.title('Loss of all theta')
    #    plt.legend()
    #    plt.xlabel('Number of iterations')
    #    plt.ylabel('Loss')
    #    plt.savefig("loss_alltheta.png")


    # all the code below this is for doing other sections of the assignment 
    elif(args.section == 3) : 
        ridge = True
        lam = 25.0
        reg = scikit_lib(X,y,ridge,lam)
        
        reg_term = lam*np.sum(np.array((reg.coef_))**2)
        val_mse = mse(reg.predict(X_val), y_val,0)
        val_mae = mae(reg.predict(X_val), y_val,0)
        print(val_mse)
        print(val_mae)
    elif(args.section == 4) :
        kbest = True
        if (kbest) : 
            X_new = sel_feature(X,y)
            X_new_val = sel_feature(X_val,y_val)
            lr = 0.001
            iters = 100000
            thresh = 0.000001
            theta,cost_train,cost_val = gradient_descent(X_new,y,lr,iters,thresh,X_new_val,y_val)

            y_train = np.reshape(y,(len(y),1))
            y_train_pred = np.c_[np.ones((len(X_new), 1)), X_new].dot(theta)
            
            train_mse = mse(y_train_pred, y_train,0)
            # train_mae = mae(y_train_pred, y_train,0)
            print(train_mse)
            # print(train_mae)

            y_val_o = np.reshape(y_val,(len(y_val),1))
            y_val_pred = np.c_[np.ones((len(X_new_val), 1)), X_new_val].dot(theta)

            val_mse = mse(y_val_pred, y_val_o,0)
            # val_mae = mae(y_val_pred, y_val_o,0)
            print(val_mse)
            # print(val_mae)

            visualisation(cost_train,cost_val,"6_7_lr0.001_0.000001")

        else : 
            X_new = sel_model(X,y)
            X_new_val = sel_model(X_val,y_val)
            lr = 0.001
            iters = 10000
            thresh = 0.0001
            theta,cost_train,cost_val = gradient_descent(X_new,y,lr,iters,thresh,X_new_val,y_val)

            y_train = np.reshape(y,(len(y),1))
            y_train_pred = np.c_[np.ones((len(X_new), 1)), X_new].dot(theta)
            
            train_mse = mse(y_train_pred, y_train,0)
            train_mae = mae(y_train_pred, y_train,0)
            print(train_mse)
            print(train_mae)

            y_val_o = np.reshape(y_val,(len(y_val),1))
            y_val_pred = np.c_[np.ones((len(X_new_val), 1)), X_new_val].dot(theta)

            val_mse = mse(y_val_pred, y_val_o,0)
            val_mae = mae(y_val_pred, y_val_o,0)
            print(val_mse)
            print(val_mae)

            visualisation(cost_train,cost_val,"4_2_rt0.0001")
    else :
        # a = [1.1966835658430395,0.39611802911829247,0.292391937478048,0.38821552488693567]
        # b = [1.6583852342715832,1.9225213398503496,6.7882553286503065,0.5634606109343059]
        # c = [10,100,1000,2048]
        # plt.semilogx()
        # plt.plot(c,a,label='training data')
        # plt.plot(c,b,label='validation data')
        # # plt.plot(np.arange(1 ,10),cost_tot,label='Total MSE')
        
        # plt.title('MSE vs No.of features')
        # plt.legend()
        # plt.xlabel('No. of Features')
        # plt.ylabel('MSE')
        # plt.savefig(f'6_8_feat.png')
        EIN = [0.753258779563091,0.6623731519035474,0.8278669922884032,0.328686241861344]
        EOUT = [1.2943783794762305,1.1051858510102353,1.1505223938638538,1.8626668814053295]
       
        tf = pd.read_csv("100_d_train.csv", header=None)
        tstf = pd.read_csv("100_d_test.csv", header=None)
        X = tf.values[:,:100]
        y = tf.values[:,100]

        X_test =  tstf.values[:,:100]
        y_test = tstf.values[:,100]


        lr = 0.01
        iters = 100000
        thresh = 0.00009
        theta,cost_train,cost_val = gradient_descent(X,y,lr,iters,thresh,X,y)

        # theta,cost_train,cost_val = ridge_reg(X1,y1,lr,iters,25,thresh,X_val,y_val)
        y_train = np.reshape(y,(len(y),1))
        y_train_pred = np.c_[np.ones((len(X), 1)), X].dot(theta)
        E_in = mean_squared_error(y_train_pred,y_train)
        print(E_in)
        y_test_o = np.reshape(y_test,(len(y_test),1))
        y_test_pred = np.c_[np.ones((len(X_test), 1)), X_test].dot(theta)

        E_out = mean_squared_error(y_test_pred,y_test_o)
        print(E_out)
        c = [2,5,10,100]
        # plt.plot(c,b,label='validation data')
        # # plt.plot(np.arange(1 ,10),cost_tot,label='Total MSE')
        
        # plt.title('MSE vs No.of features')
        # plt.legend()
        # plt.xlabel('No. of Features')
        # plt.ylabel('MSE')
        # plt.savefig(f'6_8_feat.png')
        # # visualisation(cost_train,cost_val,"2_dcsv")
    