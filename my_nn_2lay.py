import numpy as np
import matplotlib.pyplot as plt
import math
# Зерно для генератора
np.random.seed(42)
bias_val = 1
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
        self.learning_rate = 0.001
        self.init_net()
    def init_net(self):
        self.in_layer1 = np.random.randn(2,3)*math.sqrt(2.0/2)
        self.in_layer2 = np.random.randn(3,1)*math.sqrt(2.0/3)
    def direct_motion(self, _vector):
        self.e1 = np.dot(self.in_layer1, _vector.T)
        self.hidden1 = self.operations(RELU,self.e1)
        self.e2 = np.dot(self.in_layer2, self.hidden1)
        self.hidden2 = self.operations(RELU,self.e2)
        return self.hidden2
    def plot_history(x,y):
        fig, ax = plt.subplots()
        plt.plot()
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
        # print("out %s",self.show_matrix_hash(out_NN)[0])
        """
             Расчет ошибки 
        """
        delta_out = abs(target.T - out_NN)
        grads_out_nn = delta_out * self.operations(RELU_DERIV,self.e2)
        grads_on_lay2 = np.dot(grads_out_nn.T, self.in_layer2) * self.operations(RELU_DERIV,self.e1.T)# Подправлям дифференциалы
        grads_on_lay1 = np.dot(grads_on_lay2, self.in_layer1)  # Подправлям дифференциалы
        """
           Коррекция ошибки
        """
        self.in_layer1 += (vector * grads_on_lay1 * l_r)
        self.in_layer2 += (self.e1.T * grads_on_lay2 * l_r)
        return self.calculate_minimal_square_error(delta_out)
    def learn(self,init_l_r, epocha: int, train_set, target_set):
        error = 0.0
        iteration: int = 0
        n_epochs = []
        n_mse = []
        while (iteration < epocha):
            for i in range(train_set.shape[0]):
                single_2D_array_train = np.array([train_set[i]])
                single_2D_array_target = np.array([target_set[i]])
                error = self.back_propagate(single_2D_array_train, single_2D_array_target,init_l_r)
            if iteration % 1 == 0:
                print(error)
                n_epochs.append(iteration)
                n_mse.append(error)
            iteration += 1
    def operations(self,op=0,a=0,b=0,c=0,d=0,str=""):
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
def create_nn() -> NN:
    return NN()