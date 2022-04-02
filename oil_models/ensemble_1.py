import numpy as np

from oil_models.base_ensemble import BaseEnsemble
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from oil_models.utils import approximate_oil


class LinearAndExpEnsemble(BaseEnsemble):
    """
    Объединение линейной и экспоненциальной аппроксимации для предсказания ВНФ в зависимости от накопленной нефти. 
    """
    
    def __init__(self, oil_train, liq_train, liq_test, OIZ=3000):
        self.info = "linear + exp on cumulative oil"
        self.OIZ = OIZ
        self.oil_train = oil_train
        self.liq_train = liq_train
        self.liq_test = liq_test
        self.test_size = liq_test.shape[0]

    def return_coefs(self, x, y, alpha, beta, a, b, period):
        vnf_linear = a*x[::-1][:period] + b
        vnf_exp = alpha * np.exp(beta * (x[::-1][:period] - x[::-1][0]))
        def compute_error(args):
            coef1 = args[0]
            coef2 = args[1]
            return np.sum(np.power((vnf_linear * coef1 + vnf_exp * coef2) - y[::-1][:period], 2))
        
        linear_constraint = LinearConstraint([1, 1], [1], [1])
        x0 = np.array([0.5, 0.5])
        bounds = Bounds([0, 0], [1.0, 1.0])
        
        res = minimize(compute_error, x0, bounds=bounds, constraints=[linear_constraint])
        return res.x


    def predict(self):
        q_oil = self.oil_train
        q_liq = self.liq_train
        period = 50
        k = 10
        for i in range(self.test_size):
            x = np.array(np.cumsum(q_oil))
            y = np.array(((q_liq - q_oil) / (q_oil + 1e-6)))
            delta = (1/(1+y[::-1][0])) * q_liq[::-1][0]
            if i == 0:
                res = approximate_oil(x, y, 100)
                print(res[3])
                x_mean = x[::-1][0]
                y_mean = y[::-1][:k].mean()
                beta = np.log(49 / y[::-1][:k].mean()) / self.OIZ
                alpha = y[::-1][:k].mean() 
                print('alpha - ', alpha)
                print('beta - ', beta)
            else:
                x_mean = x[::-1][0]
                y_mean = y[::-1][0]
            a = res[2][0]
            b = (-a*x_mean + y_mean)
            next_VNF_linear = a*(x[::-1][0] + delta) + b
            #print(next_VNF_linear)
            next_VNF_exp = alpha * np.exp(beta * (x[::-1][0] - x[::-1][i]))
            #print(next_VNF_exp)
            if i == 0:
                coef1, coef2 = self.return_coefs(x, y, alpha, beta, a, b, period)
                print('coef linear - ', coef1)
                print('coef exp - ', coef2)
            next_VNF = coef1 * next_VNF_linear + coef2 * next_VNF_exp
            #print(next_VNF)
            q_liq_next = self.liq_test[i]
            q_oil_next = (1/(next_VNF + 1)) * q_liq_next
            q_oil = np.array(list(q_oil) + list([q_oil_next]))
            q_liq = np.array(list(q_liq) + list([q_liq_next]))
        return q_oil, coef1, coef2
