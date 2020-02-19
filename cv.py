import numpy as np

class NN_pars:
    inputNeurons=None
    outputsNeurons=None
NN=NN_pars()
NN.inputNeurons=2
NN.outputNeurons=1
max_in_nn=2
rows_orOut=1
validSet_rows=4
# 'Лентирование' матриц
def get_x_data(x,in_:int,rows:int):
    vec_x = [0] * in_ * rows
    assert rows!=0
    for r in range(rows):
        for e in range(in_):
            val=x[r][e]
            vec_x[r*NN.inputNeurons+e]=val
    return vec_x
# 'Лентирование' матриц
def get_y_data(y,out:int,rows:int):
    assert rows!=0
    vec_y = [0] * out * rows
    assert rows!=0
    for r in range(validSet_rows):
        for e in range(NN.outputNeurons):
            val=y[r][e]
            vec_y[r*NN.outputNeurons+e]=val
    return vec_y
def cross_validation(obj_nn: object, X_test: object, Y_test: object, rows: object, cols_X_test: object, cols_Y_test: object)-> object:
    """
    Производит (кросс-валидацию) предсказание и сверка ответов
    по всему  Обучающий набор/тестовая выборка
    :param X_test: 2D Обучающий набор/тестовая выборка
    :param Y_test: 2D Набор ответов/тестовая выборка
    :param rows: количество рядов в этих наборах
    :param cols_X_test: сколько элементов в обучающем наборе/тестовая выборка
    :param cols_Y_test: сколько элементов в наборе ответов/тестовая выборка
    :return: аккуратность в процентах
    """
    assert type(X_test)==list
    assert type(Y_test)==list
    tmp_vec_x_test=[0]*max_in_nn
    tmp_vec_y_test=[0]*rows_orOut
    scores=[0]*validSet_rows
    index_row=0
    res=0
    out_NN=None
    res_acc=0
    for row in range(rows):
        for elem1 in range(NN.inputNeurons):
            tmp_vec_x_test[elem1]=X_test[row*cols_X_test+elem1]
        for elem2 in range(NN.outputNeurons):
            tmp_vec_y_test[elem2]=Y_test[row*cols_Y_test+elem2]
        # predicr в out_NN
        out_NN=obj_nn.direct_motion(np.array([tmp_vec_x_test]))
        res=check_2oneHotVecs(out_NN.tolist()[0],tmp_vec_y_test,NN.outputNeurons)
        scores[index_row]=res
        index_row+=1
    res_acc=calc_accur(scores,rows)
    print("Acсuracy:%f%s"%(res_acc,"%"))
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
