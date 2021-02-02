
from hyperopt import hp, Trials, fmin, tpe
from hyperopt.pyll import scope
from hyperopt import STATUS_OK
import optuna
import numpy as np
from time import time

from bayes_opt import BayesianOptimization
from scipy.stats import uniform, randint
from bayes_opti_utils import BayesianSearchCV as BayesianSearchCV_skopt
from sklearn.model_selection import (
    cross_val_score, RandomizedSearchCV, train_test_split, GridSearchCV
)


def opt_random(estimator, params, scoring, cv, n_iter, maximize, X, y, n_jobs=1):
    opt = {}

    t0 = time()
    opt['opt'] = RandomizedSearchCV(
        estimator, params, n_iter=n_iter, cv=cv, scoring=scoring, verbose=1, n_jobs=n_jobs)
    opt['opt'].fit(X, y)
    opt['model'] = opt['opt'].best_estimator_
    opt['best_score_'] = opt['opt'].best_score_
    opt['results'] = opt['opt'].cv_results_['mean_test_score']

    fn = np.argmax if maximize else np.argmin
    opt['idx_best_score'] = fn(opt['opt'].cv_results_['mean_test_score'])

    t = time() - t0
    opt['time'] = t

    return opt


def map_best_param(best_param, param_space):
    """Maps the parameters set by using hyperopt with
    real value (which can be used to instantiate the model)

    This function is mainly for param set with hp.choice or
    if it's an integer

    Parameters
    ----------
    best_param : dict
        Dictionnary with the best selected value
        for each parameters 
    param_space : dict
        Dictionnary with the proba distribution for
        each parameters

    Returns
    -------
    dict
        Dictionnary with the best parameters in the
        correct format to instanciate the model
    """
    for key, value in param_space.items():
        best_val = best_param[key]

        if value.name == 'switch':
            _values = value.pos_args[1:]
            best_val = _values[best_val].obj

        elif value.name == 'int':
            best_val = int(best_val)

        best_param[key] = best_val

    return best_param


def opt_hyperopt(estimator, params_search, scoring, cv, n_iter, maximize, X, y, n_jobs=1):

    def objective_function(params):
        _model = estimator.set_params(**params)
        score = cross_val_score(_model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs).mean()
        loss = -score if maximize else score

        return {'loss': loss, 'status': STATUS_OK}

    opt = {}

    t0 = time()

    trials = Trials()
    best_param = fmin(objective_function,
                      params_search,
                      algo=tpe.suggest,
                      max_evals=n_iter,
                      trials=trials,
                      rstate=np.random.RandomState(42))

    opt['trials'] = trials

    best_param = map_best_param(best_param, params_search)
    opt['best_param'] = best_param
    best_estimator = estimator.set_params(**best_param)
    best_estimator.fit(X, y)

    score = cross_val_score(best_estimator, X, y,
                            cv=cv,
                            scoring=scoring).mean()

    opt['model'] = best_estimator
    opt['best_score_'] = score
    opt['results'] = [e['loss'] for e in opt['trials'].results]

    fn = np.argmin
    opt['idx_best_score'] = fn(opt['results'])

    t = time() - t0
    opt['time'] = t

    return opt


def opt_skopt(estimator, params, scoring, cv, n_iter, maximize, X, y, n_jobs=1):

    opt = {}
    t0 = time()

    opt_ = BayesianSearchCV_skopt(
        estimator,
        params,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        maximize=maximize, 
        n_jobs=n_jobs
    )

    opt_.fit(X, y)

    opt['opt'] = opt_
    opt['model'] = opt['opt'].best_estimator_
    opt['best_score_'] = opt['opt'].best_score_
    opt['results'] = opt['opt'].results_['func_vals']

    fn = np.argmin
    opt['idx_best_score'] = fn(opt['results'])

    t = time() - t0
    opt['time'] = t

    return opt


def format_bayesopti_param(params, param_details):

    for p, v in param_details.items():
        if v == 'int':
            params[p] = int(params[p])
        elif type(v) == list:
            i = int(params[p])
            if i > len(v)-1:
                i = len(v)-1
            params[p] = v[i]

    return params


def opt_bayesianopti(estimator, param_search, param_details, scoring, cv, n_iter, maximize, X, y, n_jobs=1):

    opt = {}
    t0 = time()

    def objective_function(**params):
        params = format_bayesopti_param(params, param_details)

        _model = estimator.set_params(**params)
        score = cross_val_score(_model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs).mean()
        loss = -score if not maximize else score

        return loss

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_search,
        random_state=42,
        verbose=0
    )
    optimizer.maximize(
        init_points=2,
        n_iter=n_iter-2,
    )

    opt['opt'] = optimizer

    best_params = format_bayesopti_param(
        opt['opt'].max['params'], param_details)
    best_estimator = estimator.set_params(**best_params)
    best_estimator.fit(X, y)

    score = cross_val_score(best_estimator, X, y,
                            cv=cv,
                            scoring=scoring).mean()

    opt['model'] = best_estimator
    opt['best_score_'] = score
    opt['results'] = [e['target'] for e in opt['opt'].res]

    fn = np.argmax
    opt['idx_best_score'] = fn(opt['results'])

    t = time() - t0
    opt['time'] = t

    return opt


def opt_optuna(estimator, params_search, scoring, cv, n_iter, maximize, X, y, n_jobs=1):

    def objective_function(trial):

        params = {}
        for p, v in params_search.items():
            dtype = v[0]
            if dtype == 'int':
                params[p] = trial.suggest_int(p, v[1], v[2])
            elif dtype == 'categorical':
                params[p] = trial.suggest_categorical(p, v[1])
            elif dtype == 'float':
                params[p] = trial.suggest_float(p, v[1], v[2])
            elif dtype == 'uniform':
                params[p] = trial.suggest_uniform(p, v[1], v[2])
            elif dtype == 'loguniform':
                params[p] = trial.suggest_loguniform(p, v[1], v[2])

        _model = estimator.set_params(**params)
        score = cross_val_score(_model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs).mean()

        return score

    opt = {}

    t0 = time()

    direction = 'maximize' if maximize else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective_function, n_trials=n_iter)

    opt['opt'] = study
    best_params = opt['opt'].best_params
    best_estimator = estimator.set_params(**best_params)
    best_estimator.fit(X, y)

    score = cross_val_score(best_estimator, X, y,
                            cv=cv,
                            scoring=scoring).mean()

    opt['model'] = best_estimator
    opt['best_score_'] = score
    opt['results'] = [e.values[0] for e in opt['opt'].trials]

    fn = np.argmax if maximize else np.argmin
    opt['idx_best_score'] = fn(opt['results'])

    t = time() - t0
    opt['time'] = t

    return opt
