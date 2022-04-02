import numpy as np
import time
from scipy.optimize import minimize

class CRMP:
    def __init__(self, Vp, PI, Coef_1, Date, Inj, q0, Pwf, two_inj=False, Coef_2=0):
        self.parameters = {
            'Vp': Vp,
            'PI': PI,
            'Coef_1': Coef_1,
            'Coef_2': Coef_2,
            'Tau': 0.0004969571 * 10**6 * Vp / PI
        }
        
        mask = q0 > 0
        self.Date = Date[mask]
        if two_inj:
            self.Inj_1 = Inj[0][mask]
            self.Inj_2 = Inj[1][mask]
        else:
            self.Inj_1 = Inj[mask]
            self.Inj_2 = Inj[mask]
        self.q0 = q0[mask]
        self.Pwf = Pwf[mask]
        self.two_inj = two_inj

        

        
    def predict(self, final=False):
        well_pred = []
        n_error = []
        for n in range(1, self.q0.shape[0],1):
            if (n == self.q0.shape[0] - 1) and not final:
                error = self.q0[n]
                n_error.append(error)
                continue
            delta_days = (self.Date[n].astype('datetime64[D]') - self.Date[0].astype('datetime64[D]')).astype('int') 
            if delta_days > 0:
                Depletion = self.q0[0] * np.exp(-delta_days / self.parameters['Tau'])
                summa_exp = 0
                inj_sum = 0 
                for k in range(2,n+1,1): 
                    inj_sum = self.Inj_1[k] * self.parameters['Coef_1'] + self.Inj_2[k] * self.parameters['Coef_2']
                    if k != n:
                        delta_days_2 = (self.Date[n].astype('datetime64[D]') - self.Date[k].astype('datetime64[D]')).astype('int') 
                    else:
                        delta_days_2 = 0
                    exp_1 =  np.exp(- delta_days_2 / self.parameters['Tau'])
                    exp_2 = 1 - np.exp(-1 / self.parameters['Tau'])
                    exp_3 = inj_sum - (self.parameters['PI'] * self.parameters['Tau']) * (self.Pwf[k] - self.Pwf[k-1])
                    summa_exp +=  exp_1 * exp_2 * exp_3            
                error = (Depletion + summa_exp)
            #########
            n_error.append(error)
        well_pred.append(np.array(n_error)) 
        well_pred = np.array(well_pred)
        return well_pred
    
    
    def error_for_well(self, args):
        self.parameters['Vp'] = args[0]
        self.parameters['PI'] = args[1]
        self.parameters['Coef_1'] = args[2]
        if self.two_inj:
            self.parameters['Coef_2'] = args[3]
        pred = self.predict()
        #print(pred)
        #print(self.q0.shape)
        error = np.sum(np.abs((pred - self.q0[1:])) / self.q0[1:]) / (self.q0.shape[0] - 1)
        return error 
    
    def minimize_and_predict(self, bounds, x0):
        start = time.time()
        print('Start the minimize function....Please wait')
        res = minimize(self.error_for_well, x0, bounds=bounds)
        args = res.x
        self.parameters['Vp'] = args[0]
        self.parameters['PI'] = args[1]
        self.parameters['Coef_1'] = args[2]
        if self.two_inj:
            self.parameters['Coef_2'] = args[3]

        scipy_best_pred = self.predict().reshape((-1,))
        print('Total time = ', time.time() - start)

        return scipy_best_pred
        