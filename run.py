#%%
from dataloader import *
from implementations_lucas import * 
indexes,training,y = load_train() 


x_train,x_test,y_train,y_test = split_data(training,y,0.8,42)
x_train,_,_ = standardize(x_train)
x_test,_,_ = standardize(x_test)



#%%
# Least Squares:
def train(x,y,model,regularizer=None):
    if regularizer != None:
        optimal_w = model(y,x,regularizer)
    else:
        optimal_w = model(y,x)
    return optimal_w

def predict(weights,degree=-1):
    # Writes prediction of test data 
    indexes,data = load_test()
    data,_,_ = standardize(data)
    if degree != -1:
        data = build_poly(data,degree)
    print(data.shape)
    print(weights.shape)
    prediction = data @ weights
    f = open("Data/submission.csv","w") 
    f.write("Id,Prediction\n")
    for (index,cur_prediction) in zip(indexes,prediction):
        if cur_prediction < 0:
            predict = -1 
        else:
            predict = 1 
        f.write(f"{int(index)},{predict}\n")


# %%


methods_unregularized = [least_squares]
methods_regularized = [ridge_regression]
methods_gradients = [gradient_descent,stochastic_gradient_descent]
bestmethod = None 
bestloss = 100000
bestdegree = None 
bestbatch = None 
bestregularizer = None 
for degree in range(0,20):
    x_train_poly = build_poly(x_train,degree)
    x_test_poly = build_poly(x_test,degree)
    for index_reg,regularizer in enumerate(np.logspace(0.1,10,4)): #also using this as stepsizes
        for index_batch,batch_size in enumerate([1]):
            for method in methods_unregularized + methods_regularized + methods_gradients:
                if method in methods_unregularized:
                    if index_reg != 0 or index_batch != 0:
                        continue 
                    optimal_weights = train(x_train_poly,y_train,method)
                elif method in methods_regularized:
                    if index_batch != 0:
                        continue 
                    optimal_weights = train(x_train_poly,y_train,method,regularizer)
                if method in methods_gradients:
                    inital_w = np.zeros(x_train_poly.shape[1])
                    if method in [stochastic_gradient_descent]:
                        optimal_weights = stochastic_gradient_descent(y_train,x_train_poly,inital_w,batch_size,200,regularizer)[1][-1]
                    elif index_batch == 0:
                        optimal_weights = gradient_descent(y_train,x_train_poly,inital_w,200,regularizer)[1][-1]
                    else:
                        continue 
                val_loss = compute_mse(y_test,x_test_poly,optimal_weights)
                print("Validation loss:", val_loss)
                if bestloss > val_loss:
                    bestloss = val_loss
                    bestdegree = degree
                    bestmethod = method 
                    bestregularizer = regularizer
                    bestbatch = batch_size
                    best_weights = optimal_weights

predict(best_weights,bestdegree)
# %%
print("Done")
# %%
