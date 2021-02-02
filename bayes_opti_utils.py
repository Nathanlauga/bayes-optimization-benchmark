from skopt import gp_minimize, Optimizer
from skopt.plots import plot_evaluations, plot_objective, plot_convergence
from skopt.plots import plot_convergence
from skopt.space.space import Real, Integer, Categorical
from skopt.utils import use_named_args

import math
import copy
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_random_state


def dimensions_aslist(search_space):
    """Convert a dict representation of a search space into a list of
    dimensions, ordered by sorted(search_space.keys()).
    Parameters
    ----------
    search_space : dict
        Represents search space. The keys are dimension names (strings)
        and values are instances of classes that inherit from the class
        :class:`skopt.space.Dimension` (Real, Integer or Categorical)
    Returns
    -------
    params_space_list: list
        list of skopt.space.Dimension instances.
    Examples
    --------
    >>> from skopt.space.space import Real, Integer
    >>> from skopt.utils import dimensions_aslist
    >>> search_space = {'name1': Real(0,1),
    ...                 'name2': Integer(2,4), 'name3': Real(-1,1)}
    >>> dimensions_aslist(search_space)[0]
    Real(low=0, high=1, prior='uniform', transform='identity')
    >>> dimensions_aslist(search_space)[1]
    Integer(low=2, high=4, prior='uniform', transform='identity')
    >>> dimensions_aslist(search_space)[2]
    Real(low=-1, high=1, prior='uniform', transform='identity')
    """
    params_space_list = [
        search_space[k] for k in sorted(search_space.keys())
    ]
    return params_space_list

class BayesianSearchCV():
    """Find best parameters from a given param_space using
    Bayesian optimization (package hyperopt)

    For the scoring method : 
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    Parameters
    ----------
    model :
        Model class (not instantiate)
    param_space : dict
        Dictionnary with the proba distribution for
        each parameters
    X_train : array-like 
        Feature to train the model
    y_train : array-like
        Label to train the model
    n_iter : int
        Number of maximum iteration
    cv : int, optional
        Determines the cross-validation splitting strategy, by default 5
    scoring : str, callable, list/tuple or dict, default=None
        A single str or a callable to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
        If None, the estimator's score method is used.
    maximize : bool, optional
        Whether you want to maximize the score (e.g. accuracy)
        or to minimize it (e.g. RMSE), by default True

    Returns
    -------
    best_model
        The model fitted with the best hyperparameters
    """
    
    opt = None
    
    def __init__(self,
                 estimator,
                 params_space,
                 n_iter,
                 n_jobs=None, cv=5, scoring=None, maximize=True, random_state=None, 
                 init_params=None,
                 exploration_decay=None,
                 exploration_min=None,
                 **optimizer_kwargs_):
        """
        """
        self.estimator = estimator
        self.params_space = params_space        
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.maximize = maximize
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.init_params = init_params
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.optimizer_kwargs_ = optimizer_kwargs_
        

    def _select_best_params(self):
        if self.opt is None:
            raise ValueError('BayesianSearchCV was not fitted')
            
        values = self.opt.get_result().x
        keys = [x.name for x in self.opt.get_result().space]

        params = {}

        for k, v in zip(keys, values):
            params[k] = v

        return params
    
    def plot_convergence(self):
        if self.opt is None:
            raise ValueError('BayesianSearchCV was not fitted')
            
        plot_convergence(self.results_)
        
    def plot_objective(self):
        if self.opt is None:
            raise ValueError('BayesianSearchCV was not fitted')
            
        plot_objective(self.results_)
        
    def _make_optimizer(self, params_space):
        """Instantiate skopt Optimizer class.

        Parameters
        ----------
        params_space : dict
            Represents parameter search space. The keys are parameter
            names (strings) and values are skopt.space.Dimension instances,
            one of Real, Integer or Categorical.

        Returns
        -------
        optimizer: Instance of the `Optimizer` class used for for search
            in some parameter space.

        """

        random_state = check_random_state(self.random_state)
        self.optimizer_kwargs_['random_state'] = random_state
        
        kwargs = self.optimizer_kwargs_.copy()
        
        kwargs['dimensions'] = dimensions_aslist(params_space)
        optimizer = Optimizer(**kwargs)
        for i in range(len(optimizer.space.dimensions)):
            if optimizer.space.dimensions[i].name is not None:
                continue
            optimizer.space.dimensions[i].name = list(sorted(
                params_space.keys()))[i]

        return optimizer
    
    
    def _format_param(self, params):
        """
        """
        keys = list(sorted(self.params_space.keys()))
        
        params_dict = {}
        
        for k, v in zip(keys, params):
            params_dict[k] = v
            
        return params_dict
    
    
    def _init_optimizer(self, opt, fun):
        """
        """
        if self.init_params is None:
            return opt

        for x in self.init_params:
            opt.tell(x, fun(x))

        return opt
    
    def _update_exploration_params(self, opt):
        """
        """
        acq_func_kwargs = opt.acq_func_kwargs
        print(acq_func_kwargs)
        
        if "xi" in acq_func_kwargs:
            xi = acq_func_kwargs['xi'] * self.exploration_decay
            if xi >= self.exploration_min:
                acq_func_kwargs['xi'] = xi
                
        if "kappa" in acq_func_kwargs:
            kappa = acq_func_kwargs['kappa'] * self.exploration_decay
            if kappa >= self.exploration_min:
                acq_func_kwargs['kappa'] = kappa
        
        opt.acq_func_kwargs = acq_func_kwargs
        opt.update_next()
        
        return opt
    
    def _run(self, opt, fun):
        """
        """
        # xi or kappa has to be set to use exploration decay
        xi_not_set = kappa_not_set = True
        if 'acq_func_kwargs' in self.optimizer_kwargs_:
            xi_not_set = not "xi" in self.optimizer_kwargs_['acq_func_kwargs']
            kappa_not_set = not "kappa" in self.optimizer_kwargs_['acq_func_kwargs']
        
        if (self.exploration_decay is None) | (xi_not_set & kappa_not_set):
            opt.run(fun, n_iter=self.n_iter)
            return opt
    
        n_initial_points = opt._n_initial_points
        
        # run on initial points
        opt.run(fun, n_iter=n_initial_points)
    
        # loop : each step apply exploration decay on xi or kappa acq function param
        for i in range(self.n_iter - n_initial_points): 
            opt = self._update_exploration_params(opt)
            opt.run(fun, n_iter=1)
            
        return opt
    
    
    def fit(self, X, y, **fit_params):
        """
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : array-like of shape (n_samples, n_output) or (n_samples,)
        acquisition_weight: 
        """        
        
#         @use_named_args(self.params_space)
        def objective_function(params):
            """Objective function to optimize.

            /!\ if your metric needs to be maximize please 
            /!\ use maximize=True

            Parameters
            ----------
            x: list
                List of parameters for
                the model

            Returns
            -------
            dict:
                dictionnary with loss and status
            """
            params = self._format_param(params)
            # print(params)
            
            if self.random_state is not None:
                params['random_state'] = self.random_state
            
            model = self.estimator.set_params(**params)
            score = cross_val_score(model, X, y, fit_params=fit_params,
                                    cv=self.cv, scoring=self.scoring,
                                    n_jobs=self.n_jobs, error_score='raise').mean()

            if self.maximize:
                score = -score
            # print(score)
    
            return score
        
#         opt = Optimizer(self.params_space, random_state=self.random_state, **self.optimizer_kwargs_)
        opt = self._make_optimizer(self.params_space)
        
        opt = self._init_optimizer(opt, objective_function)
    
        opt = self._run(opt, objective_function)
#         opt.run(objective_function, n_iter=self.n_iter)
        
        self.opt = opt
        self.results_ = opt.get_result()
    
        # Map best parameters
        best_param = self._select_best_params()
        
        if self.random_state is not None:
            best_param['random_state'] = self.random_state
        
        # Fit the best model
        best_estimator = self.estimator.set_params(**best_param)
        best_estimator.fit(X, y)
        
        self.best_estimator_ = best_estimator
        self.best_param_ = best_param
        self.best_score_ = cross_val_score(best_estimator, X, y, 
                                           cv=self.cv, 
                                           scoring=self.scoring).mean()
        
        return self