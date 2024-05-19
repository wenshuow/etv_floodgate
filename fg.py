from scipy.stats import bernoulli, norm
# from sklearn import linear_model
import numpy as np
from copy import deepcopy
# import math
import sys
import timeit
# import csv
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
# import xgboost as xgb
from scipy.integrate import dblquad, quad
from os import path
import xgboost as xgb
    
def cov_matrix(p, cov_type = 'equi', rho = 0.5):
    if cov_type == 'equi':
        return rho * np.ones((p, p)) + (1-rho) * np.diag(np.ones(p))
    if cov_type == 'AR1':
        return np.array([[rho**abs(i-j) for j in range(p)] for i in range(p)])
    raise ValueError

def exp_ratio(num, denom):
    log_max = np.maximum(denom, num)
    return np.exp(num-log_max)/(np.exp(denom-log_max)+np.exp(num-log_max))

class probit_oracle():
    # oracle predictor for E when underlying model is probit
    def __init__(self, beta, var_index, Sigma):
        self.beta = beta.copy()
        self.theta = beta.copy()
        self.theta[var_index] = 0.0
        self._var_index = var_index
        self._Sigma = Sigma.copy()
        # print(self.theta, self.beta)
    def get_probs(self, cov, y):
        if self.beta[self._var_index] == 0:
            self.probs = 0.5*np.ones(cov.shape[0])
        else:
            Z_dim = self.beta.shape[0]-1
            self.probs = np.zeros(cov.shape[0])
            mu, var = mean_zero_conditional_parameters(self._Sigma, self._var_index, cov[:, 0:(Z_dim+1)])
            # print(self.beta[self._var_index] * mu)
            mean_fake = (cov.dot(self.theta) + self.beta[self._var_index] * mu)/np.sqrt(1+self.beta[self._var_index]**2*var)
            mean_true = cov.dot(self.theta)+self.beta[self._var_index]*cov[:, self._var_index]
            prob_fake_log = norm.logcdf(mean_fake)
            prob_log = norm.logcdf(mean_true)
            if1 = exp_ratio(prob_log, prob_fake_log)
            prob_sf_fake_log = norm.logsf(mean_fake)
            prob_sf_log = norm.logsf(mean_true)
            if0 = exp_ratio(prob_sf_log, prob_sf_fake_log)
            for i in range(cov.shape[0]):
                self.probs[i] = if1[i] if y[i] == 1 else if0[i]
    def predict(self, cov, y, threshold):
        self.get_probs(cov, y)
        #print(baseline, full, self._probs)
        return np.array([1*(p>0.5) if abs(p-0.5) > abs(threshold-0.5) else 0.5 for p in self.probs])
    def predict_proba(self, cov, y):
        self.get_probs(cov, y)
        return np.concatenate((1-self.probs.reshape(-1,1), self.probs.reshape(-1,1)),
                              axis = 1)

def mean_zero_conditional_parameters(cov, j, x, return_parameter = False):
    p = cov.shape[0]
    sigmaminusj = np.linalg.pinv(np.delete(np.delete(cov, j, 0), j, 1))
    sigmajminusj = np.delete(cov, j, 0)[:, j]
    if return_parameter:
        return sigmajminusj.reshape(1, p-1) @ sigmaminusj, cov[j,j] -  (sigmajminusj.reshape(1, p-1) @ sigmaminusj @ sigmajminusj.reshape(p-1, 1))[0, 0]
    assert x.shape[1] == p, f'x must be a matrix and have {p} columns but x is {x}'
    assert not np.isnan((sigmajminusj.reshape(1, p-1) @ sigmaminusj @ np.delete(x, j, 1).T)[0, :]).any(), f"{sigmajminusj.reshape(1, p-1), sigmaminusj, np.delete(x, j, 1).T}"
    return (sigmajminusj.reshape(1, p-1) @ sigmaminusj @ np.delete(x, j, 1).T)[0, :], cov[j,j] -  (sigmajminusj.reshape(1, p-1) @ sigmaminusj @ sigmajminusj.reshape(p-1, 1))[0, 0]


class floodgate_data():
    def __init__(self, n, p, J, beta, Sigma, seed = 12, dgp = 'logistic'):
        np.random.seed(seed)
        assert dgp in ['logistic', 'probit'], f'only logistic and probit are supported now, not {dgp}'
        self.X = np.random.multivariate_normal(np.zeros(p), Sigma, size = n)
        self.beta = beta.copy()
        if dgp == 'logistic':
            odds = np.exp(self.X@beta)
            probs = odds/(odds + 1)
        if dgp == 'probit':
            probs = norm.cdf(self.X@beta)
        self.dgp = dgp
        self.Y = np.array([bernoulli.rvs(prob, size=1)[0] for prob in probs])
        self.E = np.ones(n)
        self.X_tils = []
        self._Sigma = Sigma.copy()
        self.use_weights = True
        self.J = J
        self.n = n
        self.p = p
        self.Y = self.Y.reshape((len(self.Y), 1))
        self.fg_data_ready = False
        # self.interaction = add_interaction

    def get_subset(self, subset):
        self.n = len(subset)
        self.X = self.X[subset, :]
        self.Y = self.Y[subset, :]
        self.E = self.E[subset]
        self._var_index = None

    def get_fg_data(self, var_index):
        # assert var_index in (0, 1)
        self._var_index = var_index
        # self.X_baseline = self.X[:, [1-var_index]]
        
    def resample(self, seed):
        np.random.seed(seed)
        X_til = self.X.copy()
        conditional_mean, conditional_variance = mean_zero_conditional_parameters(self._Sigma, self._var_index, self.X)
        assert not np.isnan(conditional_mean).any(), f"{conditional_mean}"
        X_til[:, self._var_index] = conditional_mean + np.sqrt(conditional_variance) * np.random.normal(size=self.n)
        return X_til
    
    def fit(self):
        self._full_model = LogisticRegression(random_state=0,
                fit_intercept = True).fit(self.X, self.Y.reshape(-1))
        self._baseline_model = LogisticRegression(random_state=0,
                fit_intercept = True).fit(self.X_baseline, self.Y.reshape(-1))
    def fit_CV(self, working_model, fold):
        self._models = []
        for icv in range(fold):
            subset, _ = get_cv_partitions(self.n, fold, icv)
            if working_model == 'oracle':
                assert self.dgp == 'probit', f'oracle model only exists for probit, input is {self.dgp}'
                self._models.append([probit_oracle(self.beta, j, self._Sigma) for j in range(self.p)])
            elif working_model == 'logistic':
                # print(icv)
                self._full_model = LogisticRegression(random_state=0,
                        fit_intercept = True).fit(self.X[subset, :], self.Y.reshape(-1)[subset])
                self._baseline_models = [LogisticRegression(random_state=0,
                        fit_intercept = True).fit(np.delete(self.X[subset, :], j, 1), self.Y.reshape(-1)[subset]) for j in range(self.p)]
                self._models.append([likelihood_model(j, base, self._full_model) for j, base in enumerate(self._baseline_models)])
            elif working_model == 'RF':
                self._full_model =RandomForestClassifier(max_depth=2, random_state=0).fit(self.X[subset, :], self.Y.reshape(-1)[subset])
                self._baseline_models = [RandomForestClassifier(max_depth=2, random_state=0).fit(np.delete(self.X[subset, :], j, 1), self.Y.reshape(-1)[subset]) for j in range(self.p)]
                self._models.append([likelihood_model(j, base, self._full_model) for j, base in enumerate(self._baseline_models)])
            else:
                raise ValueError(f"{working_model} is not a supported model")

class likelihood_model():
    def __init__(self, j, baseline_model, full_model):
        self._base = baseline_model
        self._full = full_model
        self.j = j
    def predict(self, cov, y, threshold):
        baseline = self._base.predict_proba(np.delete(cov, self.j, 1))[:, 1]
        full = self._full.predict_proba(cov)[:, 1]
        self._probs = abs(1-y-full)/(abs(1-y-full)+abs(1-y-baseline))
        #print(baseline, full, self._probs)
        return np.array([1*(p>0.5) if abs(p-0.5) > abs(threshold-0.5) else 0.5 for p in self._probs])
    def predict_proba(self, cov, y):
        baseline = self._base.predict_proba(np.delete(cov, self.j, 1))[:, 1]
        full = self._full.predict_proba(cov)[:, 1]
        self._probs = abs(1-y-full)/(abs(1-y-full)+abs(1-y-baseline))
        return np.concatenate((1-self._probs.reshape(-1,1), self._probs.reshape(-1,1)),
                              axis = 1)

def get_fg_predictions(e_model, X, y, threshold):
    return e_model.predict(X, y, threshold)
        
def get_loss(e_model, fg_data, distances, J, seed = 12):
    np.random.seed(seed)
    losses = np.zeros((fg_data.n, len(distances)))
    #     print(e_model.predict_proba(fg_data.X, fg_data.Y.reshape(-1)))
    #     pass
    for ic, c in enumerate(distances):
        losses[:, ic] += 1-get_fg_predictions(e_model, fg_data.X, fg_data.Y.reshape(-1), c)
    for k in range(J):
        X_resampled = fg_data.resample(seed = seed*(k+1)+J)
        for ic, c in enumerate(distances):
            losses[:, ic] += get_fg_predictions(e_model, X_resampled, fg_data.Y.reshape(-1), c)/J
    return losses

def get_cv_partitions(n, fold, icv):
    assert 0 <= icv < fold
    num_per_set = int(n/fold)
    sep = [x*num_per_set for x in range(fold)] + [n]
    validation_set = list(range(sep[icv], sep[icv+1]))
    training_set = [i for i in range(n) if i not in validation_set]
    return training_set, validation_set

def etv_result(distances, data, J, selection, working_model, save_name,
                    fold = 2, seed = 12, true_tvs = None):
    np.random.seed(seed)
    n, p = data.n, data.p
    # num_per_set = int(n/fold)
    # sep = [x*num_per_set for x in range(fold)] + [n]
    # var_index = x_index
    data_for_validation = deepcopy(data)
    start = timeit.default_timer()
    if selection == 'Oracle':
        means, variances = np.zeros((len(distances), p)),np.zeros((len(distances), p))
        # means_large_sample = np.zeros((len(distances), p))
        result = [] # np.zeros((len(distances), 10))
    else:
        means, variances = 0.0, 0.0
        means_large_sample = 0.0
        result = np.zeros((1, 10))
    data.fit_CV(working_model, fold)
    for icv in range(fold):
        # validation_set = list(range(sep[icv], sep[icv+1]))
        # training_set = [i for i in range(n) if i not in validation_set]
        _, validation_set = get_cv_partitions(n, fold, icv)
        data_validation = deepcopy(data_for_validation)
        data_validation.get_subset(validation_set)
        for var_index in range(p):
            data_validation.get_fg_data(var_index)
            losses = get_loss(data._models[icv][var_index], data_validation, distances, J, seed*J*icv+seed)
            means[:, var_index] += np.sum(losses, axis = 0)/n
            variances[:, var_index] += np.var(losses, axis = 0)/fold
            # if true_tvs[var_index]==0:
            #     print(var_index, losses)
    time = timeit.default_timer() - start
    for var_index in range(p):
        res = np.zeros((len(distances), 10)) if selection == 'Oracle' else np.zeros((1, 10))
        Rn_hat = means[:, var_index] # np.sum(means, axis = 0)
        sigma_n2 = variances[:, var_index] # np.sum(variances, axis = 0)
        # large_sample_mean = 2*(1 - np.mean(means_large_sample))
        fg_lower_bound = 2*(1 - Rn_hat - 1.644854*np.sqrt(sigma_n2)/np.sqrt(n))
        fg_lower_bound -= 1e-14*(sigma_n2 <= 1e-14)
        point_estimate = 2*(1 - Rn_hat)
        # print(Rn_hat, sigma_n2, fg_lower_bound, 1-Rn_hat, Rn_hat[0])
        if selection == 'Oracle':
            for ic, c in enumerate(distances):
                res[ic, :] = np.array([n, p, J, Rn_hat[ic],
                        sigma_n2[ic], fg_lower_bound[ic], max(0, fg_lower_bound[ic]), c, point_estimate[ic], time])
        else:
            res[0, :] = np.array([n, p, J, Rn_hat,
                    sigma_n2, fg_lower_bound, max(0, fg_lower_bound), c, point_estimate, time])
        res = pd.DataFrame(res, columns = ['n', 'p', 'J', 'mean', 'var', 'fg', 'fg_nonneg', 'c', 'point_est', 'time'],
                 index = None)
        res['beta_j'] = [data.beta[var_index]] * len(distances)
        if true_tvs is not None:
            res['true_TV'] = [true_tvs[var_index]] * len(distances)
            res['covered'] = 1*(fg_lower_bound <= true_tvs[var_index])
        res['j'] = [var_index] * len(distances)
        result.append(res)
    # pd.concat(result).to_csv(f'details/{save_name}_{J}_{selection}_{fold}_{seed}.csv')
    return pd.concat(result)

def get_true_TV(beta, j, Sigma):
    if beta[j] == 0:
        return 0.0
    mu, var = mean_zero_conditional_parameters(Sigma, j, None, True)
    theta = np.delete(beta, j) + beta[j] * mu[0, :]
    theta2root = np.sqrt((theta.reshape(1, -1) @ np.delete(np.delete(Sigma, j, 0), j, 1) @ theta.reshape(-1, 1))[0,0])
    beta2 = beta[j]**2 * var
    if theta2root == 0:
        # print('a')
        return 2 * quad(lambda x: np.abs(norm.cdf(x)-0.5)*np.exp(-x**2/(2*beta2))/(np.sqrt(2*np.pi*beta2)), -np.inf, np.inf)[0]
    res = 2* dblquad(lambda x, y: np.abs(norm.cdf(np.sqrt(beta2)*x+theta2root*y)-norm.cdf(theta2root*y/np.sqrt(1+beta2)))*np.exp(-(x**2+y**2)/2)/(2*np.pi), -np.inf, np.inf, -np.inf, np.inf)[0]
    return res

class floodgate_data2():
    def __init__(self, n, p, J, beta, Sigma, seed = 12, dgp = 'logistic'):
        np.random.seed(seed)
        assert len(beta) == p-1
        assert p > 1
        assert dgp in ['logistic', 'probit'], f'only logistic and probit are supported now, not {dgp}'
        self.X = np.random.multivariate_normal(np.zeros(p-1), Sigma, size = n)
        self.X = np.concatenate([self.X, bernoulli.rvs(0.5, size=n).reshape(-1, 1)], axis = 1)
        self.beta = beta.copy()
        if dgp == 'logistic':
            odds = np.exp(self.X[:, 1:(p-1)]@self.beta[1:] + self.X[:, 0]*self.X[:, p-1]*self.beta[0])
            probs = odds/(odds + 1)
        if dgp == 'probit':
            probs = norm.cdf(self.X[:, 1:(p-1)]@self.beta[1:] + self.X[:, 0]*self.X[:, p-1]*self.beta[0])
        self.dgp = dgp
        self.Y = np.array([bernoulli.rvs(prob, size=1)[0] for prob in probs])
        self.E = np.ones(n)
        self.X_tils = []
        self._Sigma = Sigma.copy()
        self.use_weights = True
        self.J = J
        self.n = n
        self.p = p
        self.Y = self.Y.reshape((len(self.Y), 1))
        self.fg_data_ready = False
        # self.interaction = add_interaction

    def get_subset(self, subset):
        self.n = len(subset)
        self.X = self.X[subset, :]
        self.Y = self.Y[subset, :]
        self.E = self.E[subset]
        
    def resample(self, seed):
        np.random.seed(seed)
        X_til = self.X.copy()
        conditional_mean, conditional_variance = mean_zero_conditional_parameters(self._Sigma, 0, self.X[:, 0:(self.p-1)])
        assert not np.isnan(conditional_mean).any(), f"{conditional_mean}"
        X_til[:, 0] = conditional_mean + np.sqrt(conditional_variance) * np.random.normal(size=self.n)
        return X_til
    
    def fit_CV(self, working_model, fold):
        self._models = []
        for icv in range(fold):
            subset, _ = get_cv_partitions(self.n, fold, icv)
            # if working_model == 'oracle':
            #     assert self.dgp == 'probit', f'oracle model only exists for probit, input is {self.dgp}'
            #     self._models.append([probit_oracle(self.beta, j, self._Sigma) for j in range(self.p)])
            if working_model == 'logistic':
                # print(icv)
                self._full_model = LogisticRegression(random_state=0,
                        fit_intercept = True).fit(self.X[subset, :], self.Y.reshape(-1)[subset])
                self._baseline_model = LogisticRegression(random_state=0,
                        fit_intercept = True).fit(np.delete(self.X[subset, :], 0, 1), self.Y.reshape(-1)[subset])
                self._models.append(likelihood_model(0, self._baseline_model, self._full_model))
            elif working_model == 'logistic_int':
                self._full_model = logistic_int_model(LogisticRegression(random_state=0,
                        fit_intercept = True).fit(np.concatenate([self.X[subset, :], self.X[subset, :]*self.X[subset, :][:, [self.p-1]]], axis = 1), self.Y.reshape(-1)[subset]))
                delete_x1 = np.delete(self.X[subset, :], 0, 1)
                self._baseline_model = logistic_int_model(LogisticRegression(random_state=0,
                        fit_intercept = True).fit(np.concatenate([delete_x1, delete_x1*delete_x1[:, [self.p-2]]], axis = 1), self.Y.reshape(-1)[subset]))
                self._models.append(likelihood_model(0, self._baseline_model, self._full_model))

class logistic_int_model():
    def __init__(self, logistic_model):
        self._model = logistic_model
    def predict(self, cov):
        return self._model.predict(np.concatenate([cov, cov * cov[:, [-1]]], axis = 1))
    def predict_proba(self, cov):
        return self._model.predict_proba(np.concatenate([cov, cov * cov[:, [-1]]], axis = 1))

def CV_select_c(fg_data, distances, working_model, selection = 'CV', fold = 10, seed = 11):
    n, p, J = fg_data.n, fg_data.p, fg_data.J
    if selection == 'Aggressive':
        return [0.5] * 1
    assert sorted(list(distances)) == list(distances)
    np.random.seed(seed)
    means, variances = np.zeros((len(distances), 1)), np.zeros((len(distances), 1))

    data_for_validation = deepcopy(fg_data)
    fg_data.fit_CV(working_model, fold)
    for icv in range(fold):
        _, validation_set = get_cv_partitions(n, fold, icv)
        data_validation = deepcopy(data_for_validation)
        data_validation.get_subset(validation_set)
        losses = get_loss(fg_data._models[icv], data_validation, distances, J, seed*J*icv+seed)
        means[:, 0] += np.sum(losses, axis = 0)/n
        variances[:, 0] += np.var(losses, axis = 0, ddof = 1)/fold
    if selection == 'CV':
        fg_bounds = 2*(1 - means - 1.644854*np.sqrt(variances)/np.sqrt(n))
        return [distances[np.argmax(fg_bounds[:, 0])]]
    else:
        raise ValueError

def Oracle_select_c(p, beta, dgp, f_model, distances, n, validation_size, J, Sigma, seed = 11):
    bounds = np.zeros(len(distances))
    for _ in range(10):
        _data = floodgate_data2(validation_size, p, J, beta, Sigma, seed+_**2+int(np.log(1+_)), dgp=dgp)
        losses = get_loss(f_model, _data, distances, J, seed*J+seed)
        means = np.mean(losses, axis = 0)
        variances = np.var(losses, axis = 0, ddof = 1)
        bounds += 2*(1 - means - 1.644854*np.sqrt(variances)/np.sqrt(n))
    return [distances[np.argmax(bounds)]]

def etv_result2(distances, data, selection, working_model, Sigma, save_name,
                    fold = 2, seed = 12, true_tvs = None):
    np.random.seed(seed)
    n, p, J = data.n, data.p, data.J
    # num_per_set = int(n/fold)
    # sep = [x*num_per_set for x in range(fold)] + [n]
    # var_index = x_index
    data_for_validation = deepcopy(data)
    start = timeit.default_timer()
    if selection == 'Oracle':
        means, variances = np.zeros((len(distances), 1)),np.zeros((len(distances), 1))
        # means_large_sample = np.zeros((len(distances), p))
        result = [] # np.zeros((len(distances), 10))
    else:
        means, variances = np.zeros((1, 1)),np.zeros((1, 1))
        # means_large_sample = 0.0
        result = []
    data.fit_CV(working_model, fold)
    c_mean = np.zeros(1)
    for icv in range(fold):
        training_set, validation_set = get_cv_partitions(n, fold, icv)
        data_validation = deepcopy(data_for_validation)
        data_validation.get_subset(validation_set)
        if selection == 'CV':
            data_for_cv = deepcopy(data_for_validation)
            data_for_cv.get_subset(training_set)
            c_s = CV_select_c(data_for_cv, distances, working_model, selection = 'CV', fold = 10, seed = 11*(icv+2)+len(distances))
            # print(icv, c_s[0])
        elif selection == 'Naive':
            # assert len(selection.split('_')) == 2, selection
            # c_s = (0.5+float(selection.split('_')[1])) * np.ones(1)
            c_s = 0.5 * np.ones(1)
        elif selection == 'CV_oracle':
            c_s = Oracle_select_c(data_validation.p, data_validation.beta, data_validation.dgp, data._models[icv], distances, n, data_validation.n, J, Sigma, seed = 11*(icv+2)+len(distances))
            # print(icv, c_s[0])
        # elif selection == 'CV_oracle2':
        #     c_s = Oracle_select_c2(data_validation.p, data_validation.beta, data_validation.dgp, data._models[icv], distances, n, data_validation.n, J, seed = 11*(icv+2)+len(distances))
        #     # print(icv, c_s[0])
        else:
            raise ValueError
        c_mean += np.array(c_s)/fold
        losses = get_loss(data._models[icv], data_validation, distances if selection == 'Oracle' else [c_s[0]], J, seed*J*icv+seed)
        means[:, 0] += np.sum(losses, axis = 0)/n
        variances[:, 0] += np.var(losses, axis = 0, ddof = 1)/fold
    time = timeit.default_timer() - start
    var_index = 0
    res = np.zeros((len(distances), 10)) if selection == 'Oracle' else np.zeros((1, 10))
    Rn_hat = means[:, var_index] # np.sum(means, axis = 0)
    sigma_n2 = variances[:, var_index] # np.sum(variances, axis = 0)
    # large_sample_mean = 2*(1 - np.mean(means_large_sample))
    fg_lower_bound = 2*(1 - Rn_hat - 1.644854*np.sqrt(sigma_n2)/np.sqrt(n))
    fg_lower_bound -= 1e-14*(sigma_n2 <= 1e-14)
    point_estimate = 2*(1 - Rn_hat)
    # print(Rn_hat, sigma_n2, fg_lower_bound, 1-Rn_hat, Rn_hat[0])
    if selection == 'Oracle':
        for ic, c in enumerate(distances):
            res[ic, :] = np.array([n, p, J, Rn_hat[ic],
                    sigma_n2[ic], fg_lower_bound[ic], max(0, fg_lower_bound[ic]), c, point_estimate[ic], time])
    else:
        res[0, :] = np.array([n, p, J, Rn_hat[0],
                sigma_n2[0], fg_lower_bound[0], max(0, fg_lower_bound[0]), c_mean[var_index], point_estimate[0], time])
    res = pd.DataFrame(res, columns = ['n', 'p', 'J', 'mean', 'var', 'fg', 'fg_nonneg', 'c', 'point_est', 'time'],
             index = distances if selection == 'Oracle' else [selection])
    n_new = len(distances) if selection == 'Oracle' else 1
    res['beta_j'] = [data.beta[var_index]] * n_new
    if true_tvs is not None:
        res['true_TV'] = [true_tvs[var_index]] * n_new
        res['covered'] = 1*(fg_lower_bound <= true_tvs[var_index])
    res['j'] = [var_index] * n_new
    res['seed'] = [seed] * n_new
    if selection == 'Oracle':
        print(res.shape, res[res.fg_nonneg == res.fg_nonneg.max()].shape)
        res = res[res.fg_nonneg == res.fg_nonneg.max()].iloc[0:1, :]
    # if selection == 'CV_oracle':
    #     res.to_csv(f'details/binary_beta{beta[0]}_{J}_{working_model}_{rho}_{p}_{kappa}_{CV}/{selection}_n{ntimes}_{int(seed/10)}.csv')
    return res