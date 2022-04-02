import numpy as np

from oil_models.base_ensemble import BaseEnsemble
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import minimize
from oil_models.utils import approximate_oil
from scipy.optimize import curve_fit

def approximate_inj(x_init, y_init, n):
    best_error = 10**10
    best_params = 0

    def linear(x,a):
        return a*x + ( -a*x[0] + y[0])
    
    l_end = n
    for l in range(n+1, x_init.shape[0]):
        x = x_init[::-1][:l]
        y = y_init[::-1][:l]
        a, _ = curve_fit(linear, x, y, maxfev = 10**8)
        err_1 = np.mean(np.power(y - a*x - (-a*x[::-1][0] + y[::-1][0]), 2))
        if err_1 < best_error:
            best_error = err_1
            best_params = a
            l_end = l
    return best_params, l_end

class LinearExpEnsembleOnInjection(BaseEnsemble):
    """
    Объединяет LinearExpEnsemble с линейной моделью для ВНФ на накопленной закачке через ошибку по нефти.
    """
    def __init__(self, oil, oil_train, liq, liq_train, liq_test, Inj, Inj_train, Inj_test, OIZ=3000):
        self.info = ""
        self.OIZ = OIZ

        self.oil = oil
        self.oil_train = oil_train

        self.liq = liq
        self.liq_train = liq_train
        self.liq_test = liq_test

        self.Inj = Inj
        self.Inj_train = Inj_train
        self.Inj_test = Inj_test

        self.test_size = liq_test.shape[0]

    def fit(self):
        period_train = self.Inj_train.shape[0]
        q_oil = self.oil
        q_liq = self.liq

        k = 10
        OIZ = self.OIZ    #offset
        VNF_1 = []
        VNF_2 = []
        VNF_3 = []
        for i in range(period_train):     
            #находим параметры для линейной 1 и экспоненты по нефти
            x = np.array(np.cumsum(q_oil))
            y = np.array(((np.abs(q_liq - q_oil)) / (q_oil + 1e-6)))
            delta = (1/(1+y[::-1][0])) * q_liq[::-1][0]
            if i == 0:
                res = approximate_oil(x, y, 50)
                #print(res[3])
                x_mean = x[::-1][0]
                y_mean = y[::-1][:k].mean()
                beta = np.log(49 / (y[-1])) / (OIZ)
                alpha = y[-1] 
                #print('alpha - ', alpha)
                #print('beta - ', beta)
            else:
                x_mean = x[::-1][0]
                y_mean = y[::-1][0]
            a_1 = res[2][0]
            b_1 = (-a_1*x_mean + y_mean)
            next_VNF_linear_1 = a_1*(x[::-1][0] + delta) + b_1
            VNF_1.append(next_VNF_linear_1)
            #print(next_VNF_linear)
            next_VNF_exp = alpha * np.exp(beta * (x[::-1][0] - x[::-1][i]))
            VNF_2.append(next_VNF_exp)
            #print(next_VNF_exp)
            #находим параметры для линейной 2
            if i == 0:
                #approximate with minimize loss
                vnf = np.array((q_liq - q_oil) / q_oil)
                mask_vnf = vnf > 0
                #print(mask_vnf.shape)
                if mask_vnf.sum() < 1:
                    print('Error! No positive vnf for well!')
                    return -1
                x_2 = self.Inj[mask_vnf]
                y_2 = np.log(vnf[mask_vnf])
                a_2, period_train = approximate_inj(x_2, y_2, 100)
                b_2 = (-a_2*x_2[::-1][0] + y_2[::-1][0])
                #approx = a*pred_Inj_cum + b
                print('Period train - ', period_train, ' samples')
            VNF_linear_2 = np.exp((a_2*(self.Inj_train[i]) + b_2))
            VNF_3.append(VNF_linear_2)
            next_VNF = 0.33 * next_VNF_linear_1 + 0.33 * next_VNF_exp + 0.33 * VNF_linear_2
            
            #print(next_VNF)
            q_liq_next = self.liq_train[i]
            q_oil_next = (1/(next_VNF + 1)) * q_liq_next
            q_oil = np.array(list(q_oil) + list([q_oil_next]))
            q_liq = np.array(list(q_liq) + list([q_liq_next]))


        def return_coefs(VNF_1, VNF_2, VNF_3, q_oil_original, q_liq_original):

            def compute_error(args):
                coef1 = args[0]
                coef2 = args[1]
                coef3 = args[2]
                next_VNF = coef1 * VNF_1 + coef2 * VNF_2 + coef3 * VNF_3
                q_oil_pred = (1/(next_VNF + 1)) * q_liq_original
                
                error = np.mean(np.abs(q_oil_original - q_oil_pred) / q_oil_pred)
                return error
            
            linear_constraint = LinearConstraint([1, 1, 1], [1], [1])
            x0 = np.array([0.4, 0.3, 0.3])
            bounds = Bounds([0, 0, 0], [1.0, 1.0, 1.0])
            
            res = minimize(compute_error, x0, bounds=bounds, constraints=[linear_constraint])
            return res.x

        VNF_1 = np.array(VNF_1)
        VNF_2 = np.array(VNF_2)
        VNF_3 = np.array(VNF_3)
        self.coef1, self.coef2, self.coef3 = return_coefs(VNF_1, VNF_2, VNF_3, self.oil_train, self.liq_train)
        print('coef linear - ', self.coef1)
        print('coef exp - ', self.coef2)
        print('coef linear 2 - ', self.coef3)
        return 0

    def predict(self):
        Inj_cum = np.concatenate((self.Inj, self.Inj_train))
        pred_Inj_cum = self.Inj_test
        q_oil = np.concatenate((self.oil, self.oil_train))
        q_liq = np.concatenate((self.liq, self.liq_train))
        k = 10
        OIZ = self.OIZ    #offset
        VNF_1 = []
        VNF_2 = []
        VNF_3 = []
        for i in range(self.test_size):
            #находим параметры для линейной 1 и экспоненты по нефти
            x = np.array(np.cumsum(q_oil))
            y = np.array(((q_liq - q_oil) / (q_oil + 1e-6)))
            delta = (1/(1+y[::-1][0])) * q_liq[::-1][0]
            if i == 0:
                res = approximate_oil(x, y, 50)
                #print(res[3])
                x_mean = x[::-1][0]
                y_mean = y[::-1][:k].mean()
                beta = np.log(49 / y[-1]) / (OIZ)
                alpha = y[-1] 
                #print('alpha - ', alpha)
                #print('beta - ', beta)
            else:
                x_mean = x[::-1][0]
                y_mean = y[::-1][0]
            a_1 = res[2][0]
            b_1 = (-a_1*x_mean + y_mean)
            next_VNF_linear_1 = a_1*(x[::-1][0] + delta) + b_1
            VNF_1.append(next_VNF_linear_1)
            #print(next_VNF_linear)
            next_VNF_exp = alpha * np.exp(beta * (x[::-1][0] - x[::-1][i]))
            VNF_2.append(next_VNF_exp)
            #print(next_VNF_exp)
            #находим параметры для линейной 2
            if i == 0:
                #approximate with minimize loss
                vnf = np.array((q_liq - q_oil) / q_oil)
                mask_vnf = vnf > 0
                x_2 = Inj_cum[mask_vnf]
                y_2 = np.log(vnf[mask_vnf])
                #print(x_2)
                #print(y_2)
                a_2, period_train = approximate_inj(x_2, y_2, 50)
                b_2 = (-a_2*x_2[::-1][0] + y_2[::-1][0])
                #approx = a*pred_Inj_cum + b
                print('Period train - ', period_train, ' samples')
            VNF_linear_2 = np.exp((a_2*(pred_Inj_cum[i]) + b_2))
            VNF_3.append(VNF_linear_2)
            next_VNF = float(self.coef1 * next_VNF_linear_1 + self.coef2 * next_VNF_exp + self.coef3 * VNF_linear_2)
            
            #print(next_VNF)
            q_liq_next = self.liq_test[i]
            q_oil_next = (1/(next_VNF + 1)) * q_liq_next
            q_oil = np.array(list(q_oil) + list([q_oil_next]))
            q_liq = np.array(list(q_liq) + list([q_liq_next]))
        return q_oil, self.coef1, self.coef2, self.coef3 