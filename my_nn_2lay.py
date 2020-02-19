import numpy as np
import matplotlib.pyplot as plt
import math
# Зерно для генератора
np.random.seed(42)
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
class NN:
    def __init__(self):
        self.e1 = None  # Взвешенная сумма сигналов на первой прослойке
        self.e2 = None  # Взвешенная сумма сигналов на второй прослойке
        self.hidden1 = None  # активированное состояние нейронов на первой прослойке
        self.hidden2 = None  # активированное состояние нейронов на второй прослойке
        self.init_net()
    def init_net(self):
        self.in_layer1 = np.random.randn(2,3)*math.sqrt(2.0/2)
        self.in_layer2 = np.random.randn(3,1)*math.sqrt(2.0/3)
    def direct_motion(self, _vector):
        """

        :param _vector: 2D вектор вопроса это numpy массив
        :return: вектор последнего слоя
        """
        self.e1 = np.dot(self.in_layer1, _vector.T)
        self.hidden1 = self.operations(RELU,self.e1)
        self.e2 = np.dot(self.in_layer2, self.hidden1)
        self.hidden2 = self.operations(RELU,self.e2)
        return self.hidden2
    def plot_history(x,y):
        fig, ax = plt.subplots()
        plt.plot(x,y)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Mse")
        plt.show()
    def calculate_minimal_square_error(self, val):
        return np.mean(np.square(val))
    def back_propagate(self, vector, target,l_r):
        """
           Прямой проход
        """
        out_NN = self.direct_motion(vector)

        """
             Расчет ошибки 
        """
        delta_out = ( out_NN-target.T)
        grads_out_nn = delta_out * self.operations(RELU_DERIV,self.e2)
        grads_on_lay2 = np.dot(grads_out_nn.T, self.in_layer2) * self.operations(RELU_DERIV,self.e1.T)# Подправлям дифференциалы
        grads_on_lay1 = np.dot(grads_on_lay2, self.in_layer1)  # Подправлям дифференциалы
        """
           Коррекция ошибки
        """
        self.in_layer1 -= (vector * grads_on_lay1 * l_r)
        self.in_layer2 -= (self.e1.T * grads_on_lay2 * l_r)
        return self.calculate_minimal_square_error(delta_out)

    def learn(self,init_l_r, epocha: int, train_set, target_set):
        """
        :param init_l_r: коэффициент обучения
        :param epocha: количество эпох
        :param train_set: обучающий набор это 2D матрица numpy
        :param target_set: набор ответов это 2D матрица numpy
        :return:
        """
        error = 0.0
        iteration: int = 0
        n_epochs = []
        n_mse = []
        while (iteration < epocha):
            assert train_set!=None
            for i in range(train_set.shape[0]):
                """
                Здесь извлекаем 1D numpy массив,но для сети нам нужен вектор
                с измерением 2(по сути матрица),поэтому переводим в список,т.к.
                тип list переводим в numpy матрицу
                """
                single_2D_array_train = np.array([train_set[i]])
                single_2D_array_target = np.array([target_set[i]])
                error = self.back_propagate(single_2D_array_train, single_2D_array_target,init_l_r)
            if iteration % 1 == 0:
                print(error)
                n_epochs.append(iteration)
                n_mse.append(error)
            iteration += 1
    def operations(self,op=0,a=0,b=1,c=0,d=0,str=""):
        """
        В основно для функций активаций
        :param op: 'байт-комманда'
        :param a: <>
        :param b: <>
        :param c: <>
        :param d: <>
        :param str: <>-ее параметры
        :return:
        """
        l=[]
        if op==RELU:
            for i in a:
                if (a < 0):
                     l.append(0)
                else:
                     l.append(i)
            return np.array(l).T
        elif op==RELU_DERIV:
            for i in a:
                if (i < 0):
                   l.append(0)
                else:
                    l.append(1)
            return np.array(l).T
        elif op==TRESHOLD_FUNC:
            for i in a:
               if (i < 0):
                     l.append(0)
               else:
                    l.append(1)
            return np.array(l).T
        elif op==TRESHOLD_FUNC_DERIV:
            pass # Нет производной
        elif op==LEAKY_RELU:
            for i in a:
               if (i < 0):
                    l.append(b * a)
               else:
                    l.append(i)
            return np.array(l).T
        elif op==LEAKY_RELU_DERIV:
            for i in a:
               if (i < 0):
                    l.append(b)
               else:
                  l.append(1)
            return np.array(l).T
        elif op==SIGMOID:
            return (1.0 / (1 + np.exp(b * (-a)))).T
        elif op==SIGMOID_DERIV:
            return (b * 1.0 / (1 + np.exp(b * (-a))) * (1 - 1.0 / (1 + np.exp(b * (-a))))).T
        elif op==DEBUG:
            print("%s : %f"% str, a)
        elif op==DEBUG_STR:
            print("%s"% str)
