import numpy as np
import math

def sigmoid(val):
    if val > 600:
        return 1
    elif val < -600:
        return 0
    else:
        return 1 / (1 + math.exp(-val))

def sigmoid_diff(val):
    sig = sigmoid(val)
    return sig * (1 - sig)

class WeightNetwork:
    
    def __init__(self,shape,exponent,learning_rate) -> None:
        '''
        shape:[i,o],input number of terms,output number of terms
        exponent:eg. exponent = 2, _exp = 3, input = (1,a^1,a^2)
        learning_rate:0.1?

        method:
        load_input(input)
        load_expected_output(expected_output)
        run_input()
        return_cost()
        change_weight()
        out_output()
        out_weight()
        save(path)
        load(path)
        '''
        self._input_term_num = shape[0]
        self._output_term_num = shape[1]
        self._exp = exponent + 1
        self._learning_rate = learning_rate
        self._w = np.random.rand(self._output_term_num,(self._input_term_num * self._exp))
        self._input = np.ones((self._input_term_num * self._exp),dtype=float)
        self._expected_output = np.zeros(self._output_term_num,dtype=float)
        self._output = np.zeros(self._output_term_num,dtype=float)

    def load_input(self,input):
        for i in range(len(input)):
            for exp in range(self._exp):
                self._input[i * self._exp + exp] = float(input[i]) ** exp

    def load_expected_output(self,expected_output):
            self._expected_output = expected_output.copy()

    def run_input(self):
        for ot in range(self._output_term_num):
            self._output[ot] = sigmoid(sum(self._w[ot] * self._input))

    def return_cost(self):
        return sum((self._expected_output - self._output)**2)

    def change_weight(self):
        e = self._expected_output - self._output
        for ot in range(len(self._w)):
            for it in range(len(self._w[ot])):
                self._w[ot][it] += self._learning_rate * np.sign(self._input[it]) * e[ot]

    def out_output(self):
        return self._output.copy()

    def out_weight(self):
        return self._w.copy()

    def save(self,path):
        np.save(path,self._w)

    def load(self,path):
        self._w = np.load(path)



class RecurrentNetwork:
    pass

    





