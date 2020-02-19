import numpy as np

import my_nn_2lay as nn_file
import  cv

x=[[1, 1],[1, 0],[0, 1],[0, 0] ]
y=[[1],[1],[1],[0]]

def main():
    nn=nn_file.NN()
    # создаем данные
    X=cv.get_x_data(x, 2, 4)
    Y=cv.get_y_data(y, 1, 4)
    # запустим сеть на обучение
    nn.learn(0.07, 10, np.array(x), np.array(y))
    # произведем кросс валидацию
    cv.cross_validation(nn,X, Y, 4, 2, 1)

main()
