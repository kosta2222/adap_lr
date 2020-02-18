import numpy as np
import matplotlib.pyplot as plt
import math
class NN_pars:
    inputNeurons=None
    outputsNeurons=None
NN=NN_pars()
NN.inputNeurons=2
NN.outputNeurons=1
max_in_nn=2
rows_orOut=1
validSet_rows=4
eps=10
x=[[1,1],[1,0],[0,1],[0,0] ]
y=[[1],[1],[1],[0]]
vec_x=[0]*NN.inputNeurons*validSet_rows
vec_y=[0]*NN.outputNeurons*validSet_rows
def get_x_data():
    global vec_x
    for r in range(validSet_rows):
        for e in range(NN.inputNeurons):
            val=x[r][e]
            vec_x[r*NN.inputNeurons+e]=val
def get_y_data():
    global vec_y
    for r in range(validSet_rows):
        for e in range(NN.outputNeurons):
            val=y[r][e]
            vec_y[r*NN.outputNeurons+e]=val
RELU=0
RELU_DERIV=1
SIGMOID=2
SIGMOID_DERIV=3
TRESHOLD_FUNC=4
TRESHOLD_FUNC_DERIV=5
LEAKY_RELU=6
LEAKY_RELU_DERIV=7
DEBUG=8
DEBUG_STR=9
def operations(op,a,b,c,d,str):
    if op==RELU:
        if (a < 0):
            return 0
        else:
           return a
    elif op==RELU_DERIV:
         if (a < 0):
             return 0
         else:
             return a
    elif op==TRESHOLD_FUNC:
        if (a < 0):
              return 0
        else:
            return 1
    elif op==TRESHOLD_FUNC_DERIV:
         return 0
    elif op==LEAKY_RELU:
        if (a < 0):
             return b * a
        else:
             return a
    elif op==LEAKY_RELU_DERIV:
        if (a < 0):
            return b
        else:
            return 1
    elif op==SIGMOID:
        return 1.0 / (1 + math.exp(b * (-a)))
    elif op==SIGMOID_DERIV:
         return b * 1.0 / (1 + math.exp(b * (-a))) * (1 - 1.0 / (1 + math.exp(b * (-a))))
    elif op==DEBUG:
      print("%s : %f"% str, a);
    elif op==DEBUG_STR:
       print("%s"% str)
def cross_validation(model,X_test:list,Y_test:list,rows,cols_X_test,cols_Y_test)->float:
    tmp_vec_x_test=[0]*max_in_nn
    tmp_vec_y_test=[0]*rows_orOut
    scores=[0]*validSet_rows
    index_row=0
    res=0
    out_NN=np.array((1,NN.outputNeurons))
    res_acc=0
    for row in range(rows):
        for elem1 in range(NN.inputNeurons):
            tmp_vec_x_test[elem1]=X_test[row*cols_X_test+elem1]
        for elem2 in range(NN.outputNeurons):
            tmp_vec_y_test[elem2]=Y_test[row*cols_Y_test+elem2]
        # predicr в out_NN
        out_NN=model.predict(np.array([tmp_vec_x_test]))
        res=check_2oneHotVecs(out_NN.tolist()[0],tmp_vec_y_test,NN.outputNeurons)
        scores[index_row]=res
        res=0
        index_row+=1
    res_acc=calc_accur(scores,rows)
    print("Ac:%f%s"%(res_acc,"%"))
    return res_acc
def check_2oneHotVecs(out_NN:list,vec_y_test,vec_size)->int:
    tmp_elemOf_outNN_asHot=0
    for col in range(vec_size):
        tmp_elemOf_outNN_asHot=out_NN[col]
        if (tmp_elemOf_outNN_asHot>0.5):
              tmp_elemOf_outNN_asHot=1
        else:
            tmp_elemOf_outNN_asHot=0
        if(tmp_elemOf_outNN_asHot==int(vec_y_test[col])):
          continue
        else:
          return 0
    return 1
def calc_accur(scores:list,rows)->float:
    accuracy=0
    sum=0
    for col in range(rows):
        sum+=scores[col]
    accuracy=sum/rows*100
    return accuracy
def plot_history(history):
    fig,ax=plt.subplots()
    x=range(eps)
    plt.plot(x,history.history['loss'])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mse")
    plt.show()