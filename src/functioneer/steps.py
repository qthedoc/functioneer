# MIT License
# Copyright (c) 2025 Quinn Marsh
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import numpy as np
from scipy.optimize import minimize, dual_annealing, basinhopping, OptimizeResult

from functioneer.parameter import ParameterSet, Parameter

class AnalysisStep():
    """
    Base class for analysis steps in functioneer.
    """
    def __init__(self, condition: Callable[..., bool] | None = None) -> None:
        """
        Validates condition
        """
        if condition is not None and not isinstance(condition, Callable):
            raise ValueError(f"AnalysisStep condition must be a function that returns a bool")
        
        self.condition = condition

        # self.branch_cnt = 1

    def run(self, paramset: ParameterSet) -> Tuple[ParameterSet, ...]:
        """
        Evaluate the analysis step, returning a tuple of modified ParameterSets.

        Parameters
        ----------
        paramset : ParameterSet
            The input parameter set.

        Returns
        -------
        tuple of ParameterSet
            Modified parameter sets after applying the step.

        Raises
        ------
        ValueError
            If paramset is not a valid ParameterSet.

        TODO decide if Default behavior is to COPY or PASS the paramset
        """
        if not isinstance(paramset, ParameterSet):
            raise ValueError(f"paramset is not of type ParameterSet")

        return (paramset,)
    
    def get_details(self) -> Dict[str, Any]:
        """
        Return a dictionary of step details for error reporting.

        Returns
        -------
        dict
            Key details about the step (e.g., type, parameters).
        """
        return {'type': type(self).__name__}
    
class Define(AnalysisStep):
    """
    Define AnalysisStep: Adds parameter to parameterset
    Will create new Parameter in ParameterSet if one does not already exist.
    """
    def __init__(self, param_id: str, value: Any = None, condition: Callable[..., bool] | None = None):
        super().__init__(condition)
        try:
            Parameter.validate_id(param_id)
        except ValueError as e:
            raise ValueError(f"Invalid param_id: {str(e)}") from e
        self.parameter = Parameter(param_id, value)

    def run(self, paramset: ParameterSet) -> Tuple[ParameterSet, ...]:
        super().run(paramset)
        paramset.add_param(self.parameter)
        return (paramset,)
    
    def get_details(self) -> Dict[str, Any]:
        details = super().get_details()
        details.update({
            'param_id': self.parameter.id,
            'value': self.parameter.value
        })
        return details
    
class Fork(AnalysisStep):
    """
    Fork AnalysisStep: Splits analysis into parallel branches based on provided parameter configurations.

    Each configuration is a dictionary mapping parameter IDs to values. The analysis creates a new branch
    for each configuration by updating the parameter set with the specified key-value pairs, adding new
    parameters or overwriting existing ones as needed.

    Parameters
    ----------
    configurations : list of dict
        List of parameter configurations, where each dictionary contains parameter IDs (str) as keys
        and their corresponding values. All configurations must share the same set of parameter IDs.
    condition : callable, optional
        Condition function to determine if the step should run.

    Raises
    ------
    ValueError
        If configurations is not a list of dictionaries, is empty, contains inconsistent parameter IDs,
        or includes invalid parameter IDs.
    TypeError
        If condition is provided but is not callable.

    Examples
    --------
    >>> fork = Fork([{'x': 0}, {'x': 1}, {'x': 2}])  # Single parameter fork
    >>> fork = Fork([{'x': 0, 'y': 10}, {'x': 1, 'y': 20}])  # Multi-parameter fork with configurations
    """
    def __init__(self, configurations: List[Dict[str, Any]], condition: Callable[..., bool] | None = None):
        super().__init__(condition)
        # Validate that configurations is a non-empty list of dictionaries
        if not isinstance(configurations, (list, tuple)) or not all(isinstance(config, dict) for config in configurations):
            raise ValueError("configurations must be a list or tuple of dictionaries")
        if not configurations:
            raise ValueError("configurations cannot be empty")
        
        self.configurations = list(configurations)  # Convert to list for consistency

        # Validate parameter IDs and ensure consistency across configurations
        # param_ids = set(self.configurations[0].keys())
        for config in self.configurations:
            # if set(config.keys()) != param_ids: # may turn on this optionally
            #     raise ValueError("All configurations must have the same parameter IDs")
            for param_id in config:
                try:
                    Parameter.validate_id(param_id)
                except ValueError as e:
                    raise ValueError(f"Invalid param_id '{param_id}': {str(e)}") from e

    def run(self, paramset: ParameterSet) -> Tuple[ParameterSet, ...]:
        super().run(paramset)
        next_paramsets = []
        for config in self.configurations:
            ps = copy.deepcopy(paramset)
            for param_id, value in config.items():
                ps.update_param(param_id, value)  # Update or add parameter
            next_paramsets.append(ps)
        return tuple(next_paramsets)

    def get_details(self) -> Dict[str, Any]:
        details = super().get_details()
        details.update({
            'configurations': self.configurations
        })
        return details
    
class Evaluate(AnalysisStep):
    """
    Evaluate AnalysisStep: Evaluates a provided function and updates the ParameterSet with results.

    Parameters
    ----------
    func : callable
        The function to evaluate. Output can be any type if unpack_result=False and assign_to is a string,
        or a dictionary if unpack_result=True.
    assign_to : str or iterable of str, optional
        Parameter ID(s) where the function output is stored. If a string and unpack_result=False, stores
        the entire output (any type, e.g., dict, list, scalar) under that ID. If a string and unpack_result=True,
        expects a dictionary and unpacks all keys. If a tuple/list, requires unpack_result=True and extracts
        specified keys from a dictionary output. Defaults to func.__name__ if not a lambda.
    unpack_result : bool, optional
        If True, expects a dictionary output and unpacks its keys into the ParameterSet. If assign_to is a
        tuple/list, only those keys are unpacked; otherwise, all keys are used. If False and assign_to is a string,
        stores the entire output under assign_to. Must be True when assign_to is a tuple/list. Defaults to False if None.
    condition : callable, optional
        Condition function to determine if the step should run.

    Examples
    --------
    >>> anal = AnalysisModule({'x': 0, 'y': 0})
    >>> def add(x, y):
    ...     return x + y
    >>> anal.add.evaluate(func=add, assign_to='sum')  # Stores scalar 0 in 'sum'
    >>> anal.add.evaluate(func=add)  # Stores scalar 0 in 'add'
    >>> def dict_func(x, y):
    ...     return {'sum': x + y, 'product': x * y}
    >>> anal.add.evaluate(func=dict_func, assign_to='results')  # Stores {'sum': 0, 'product': 0} in 'results'
    >>> anal.add.evaluate(func=dict_func, assign_to=['sum', 'product'], unpack_result=True)  # Stores sum=0, product=0
    >>> anal.add.evaluate(func=dict_func, unpack_result=True)  # Unpacks sum=0, product=0
    >>> anal.add.evaluate(func=lambda x, y: [x, y], assign_to='list_result')  # Stores [0, 0] in 'list_result'
    """
    def __init__(self,
            func: Callable,
            assign_to: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
            unpack_result: bool = False,
            condition: Optional[Callable] = None
        ):
        super().__init__(condition)

        # Validate func
        if not isinstance(func, Callable):
            raise ValueError("Invalid Evaluate: 'func' must be a callable")
        self.func = func

        # Handle assign_to
        if assign_to is None:
            if func.__name__ == '<lambda>':
                raise ValueError("For lambda functions, 'assign_to' must be specified")
            assign_to = func.__name__

        # Handle unpack_result
        self.unpack_result = unpack_result if unpack_result is not None else False
        if not isinstance(self.unpack_result, bool):
            raise ValueError(f"Invalid Evaluate: 'unpack_result' must be a boolean, got {type(self.unpack_result)}")

        # Validate assign_to
        if isinstance(assign_to, str):
            try:
                Parameter.validate_id(assign_to)
            except ValueError as e:
                raise ValueError(f"Invalid Evaluate: 'assign_to' is not a valid param id, got {assign_to}: {str(e)}") from e
            self.assign_to = assign_to
        elif isinstance(assign_to, (list, tuple)) and assign_to:
            try:
                Parameter.validate_id_iterable(assign_to)
            except ValueError as e:
                raise ValueError(f"Invalid Evaluate: 'assign_to' must be a valid iterable of param ids, got {assign_to}: {str(e)}") from e
            self.assign_to = tuple(assign_to)
        else:
            raise ValueError(f"Invalid Evaluate: 'assign_to' must be a string or non-empty iterable of strings, got {assign_to}")
        
        # Error if unpack_result is False and assign_to is tuple/list
        if not self.unpack_result and isinstance(self.assign_to, tuple):
            raise ValueError(
                f"Invalid Evaluate: assign_to={self.assign_to} is a tuple/list, but unpack_result=False. "
                f"Try setting unpack_result=True to explicitly indicate a dictionary output (which a tuple/list 'assign_to' expects. "
            )

    def run(self, paramset: ParameterSet) -> Tuple[ParameterSet, ...]:
        """
        Evaluates the function and updates the ParameterSet with the output.

        Parameters
        ----------
        paramset : ParameterSet
            The input parameter set.

        Returns
        -------
        tuple of ParameterSet
            A tuple containing the updated ParameterSet.

        Raises
        ------
        ValueError
            If the function output does not match the expected format for assign_to and unpack_result.
        RuntimeError
            If an error occurs during function execution.
        """
        super().run(paramset)

        # TODO might need to deepcopy params is preservation of the upper level is needed
        next_paramset = copy.deepcopy(paramset)

        try:
            output = next_paramset.call_with_kwargs(self.func)
        except Exception as e:
            raise RuntimeError(f"Error executing function {getattr(self.func, '__name__', '<lambda>')}: {str(e)}") from e
        

        # Handle dictionary output from func (unpack_result=True)
        if self.unpack_result:
            if not isinstance(output, dict):
                raise ValueError(
                    f"Evaluate: Expected dictionary output for unpack_result=True, "
                    f"got {type(output)} with value {output}"
                )
            if not output:
                raise ValueError("Evaluate: Output dictionary is empty for unpack_result=True")
            
            # Use specified keys if assign_to is a tuple/list
            if isinstance(self.assign_to, tuple):
                for key in self.assign_to:
                    if key not in output:
                        raise ValueError(
                            f"Evaluate: Output dictionary missing key '{key}' for assign_to={self.assign_to}"
                        )
                    next_paramset.update_param(key, output[key])    

            # Unpack all keys if assign_to is None or a string
            else:
                for key, value in output.items():
                    try:
                        Parameter.validate_id(key)
                    except ValueError as e:
                        raise ValueError(f"Evaluate: Output key '{key}' is not a valid param id: {str(e)}") from e
                    next_paramset.update_param(key, value)

        # Handle any output type (unpack_result=False, string assign_to)
        else:
            next_paramset.update_param(self.assign_to, output)

        return (next_paramset,)
    
    def get_details(self) -> Dict[str, Any]:
        details = super().get_details()
        details.update({
            'func': getattr(self.func, '__name__', '<lambda>'),
            'assign_to': self.assign_to
        })
        return details

class Optimize(AnalysisStep):
    """
    Optimize AnalysisStep: Minimizes or maximizes an objective function by optimizing parameters.

    Parameters
    ----------
    func : callable
        The objective function to optimize. Must return a scalar value.
    opt_param_ids : iterable of str, optional
        Parameter IDs to optimize.    
    assign_to : str, optional
        Parameter ID where the optimized objective value is stored. Defaults to func.__name__ if not a lambda.
    direction : {'min', 'max'}, optional
        Direction of optimization. Default is 'min' (minimization).
    optimizer : str or callable, optional
        Optimization method or custom optimizer function. Default is 'SLSQP'.
        - Strings: SciPy methods ('Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
          'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dual_annealing', 'basinhopping').
        - Callable: A function with signature `optimizer(func, x0, bounds=None, options=None, **kwargs) -> OptimizeResult`.
        See SciPy's minimize, dual_annealing, or basinhopping documentation for details.
    tol : float, optional
        General tolerance for convergence (used as ftol and xtol where applicable).
    bounds : dict, optional
        Dictionary mapping opt_param_ids to (min, max) tuples, e.g., {'x': (0, 1)}.
        Required for 'dual_annealing' and 'basinhopping', optional otherwise.
    options : dict, optional
        Optimizer-specific options, e.g., {'maxiter': 1000, 'disp': True, 'ftol': 1e-6}.
        Common options:
        - 'ftol': Function value tolerance.
        - 'xtol': Parameter value tolerance.
        - 'maxiter': Maximum iterations.
        - 'disp': Display convergence messages.
    condition : callable, optional
        Condition function to determine if the step should run.
    **kwargs
        Additional keyword arguments passed to the optimizer (e.g., 'jac' for gradient, 'constraints' for SLSQP).

    Examples
    --------
    >>> anal = AnalysisModule({'x': 0, 'y': 0})
    >>> def rosenbrock(x, y):
    ...     return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    >>> anal.add.optimize(func=rosenbrock, opt_param_ids=['x', 'y'], tol=1e-6,
    ...                   bounds={'x': (0, 2), 'y': (0, 2)}, options={'disp': True})
    >>> results = anal.run()

    >>> # With constraints via kwargs
    >>> anal.add.optimize(func=rosenbrock, opt_param_ids=['x', 'y'],
    ...                   optimizer='SLSQP', constraints={'type': 'ineq', 'fun': lambda x, y: x + y - 1})

    >>> # Custom optimizer example
    >>> def custom_optimizer(func, x0, **kwargs):
    ...     from scipy.optimize import minimize
    ...     return minimize(func, x0, method='BFGS', options={'gtol': 1e-6})
    >>> anal.add.optimize(func=rosenbrock, opt_param_ids=['x', 'y'], optimizer=custom_optimizer)
    """
    def __init__(self,
        func: Callable,
        opt_param_ids: Tuple[str, ...],
        assign_to: Optional[str] = None,
        direction: str = 'min',
        optimizer: Union[str, Callable] = 'SLSQP',
        tol: Optional[float] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        options: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(condition)

        # Validate func
        if not isinstance(func, Callable):
            raise ValueError(f"Invalid Optimize: 'func' is not a valid objective function: {assign_to}")
        self.func = func

        # Handle assign_to
        if assign_to is None:
            if func.__name__ == '<lambda>':
                raise ValueError("For lambda functions, 'assign_to' must be specified.")
            assign_to = func.__name__

        # Validate assign_to
        try:
            Parameter.validate_id(assign_to)
        except ValueError as e:
            raise ValueError(f"Invalid Optimize: 'assign_to' is not a valid param id sting, got {assign_to}: {str(e)}") from e
        self.assign_to = assign_to

        # Validate opt_param_ids
        try:
            Parameter.validate_id_iterable(opt_param_ids)
        except ValueError as e:
            raise ValueError(f"Invalid Optimize: 'opt_param_ids' is not a valid param id tuple, got {opt_param_ids}: {str(e)}") from e
        if not opt_param_ids:
            raise ValueError(f"Invalid Optimize: 'opt_param_ids' must contain at least one optimization parameter")
        self.opt_param_ids = tuple(opt_param_ids)

        # Validate direction
        if direction not in ['min', 'max']:
            raise ValueError(f"Invalid Optimize: 'direction' must be 'min' or 'max', got {direction}")
        self.direction = direction

        # Validate optimizer
        if not isinstance(optimizer, (str, Callable)):
            raise ValueError(f"Invalid Optimize: 'optimizer' must be a string or callable, got {type(optimizer)}")
        self.optimizer = optimizer

        # Validate tol
        if tol is not None:
            if not isinstance(tol, (int, float)) or tol <= 0:
                raise ValueError(f"Invalid Optimize: 'tol' must be a positive number, got {tol}")
        self.tol = tol

        # Validate bounds
        if bounds is not None:
            if not isinstance(bounds, dict):
                raise ValueError(f"Invalid Optimize: 'bounds' must be a dictionary")
            for key, val in bounds.items():
                if key not in opt_param_ids:
                    raise ValueError(f"Invalid Optimize: bound key '{key}' not in opt_param_ids")
                if not (isinstance(val, tuple) and len(val) == 2 and all(isinstance(v, (int, float, type(None))) for v in val)):
                    raise ValueError(f"Invalid Optimize: bound for '{key}' must be a tuple of two numbers or None")
        self.bounds = bounds

        # Options can be any dict, default to empty dict if None
        self.options = options or {}
        # if tol is not None:
        #     self.options.setdefault('ftol', tol)  # Default tol to ftol if not specified
        #     self.options.setdefault('xtol', tol)  # Also set xtol for methods that use it

        # Store kwargs
        self.kwargs = kwargs
        
    def run(self, paramset: ParameterSet) -> Tuple[ParameterSet, ...]:
        """Run the optimization step on the given parameter set."""
        
        # Validate
        super().run(paramset)

        # Copy
        next_paramset = copy.deepcopy(paramset)

        # Create objective wrapper
        def objective_wrapper(x):
            '''
            Evaluates the objective function using optimizer values from x mapped to the objective kwargs.
            '''
            try:
                test_paramset = copy.deepcopy(next_paramset)
                test_paramset.update_param_values(dict(zip(self.opt_param_ids, x))) # Create test parameter set with values from x
                obj_val = test_paramset.call_with_kwargs(self.func) # evaluate objective parameter
                if not isinstance(obj_val, (int, float, np.number)):
                    raise ValueError(f"Objective function returned non-scalar value: {obj_val}")
                if not np.isfinite(obj_val):
                    raise ValueError(f"Objective function returned non-finite value: {obj_val}")
                return obj_val if self.direction == 'min' else -obj_val
            except Exception as e:
                raise RuntimeError(f"Error evaluating objective function: {str(e)}") from e
        
        # Initial parameters (x0)
        try:
            x0 = np.array([next_paramset[id].value for id in self.opt_param_ids], dtype=float)
            if not all(np.isfinite(x0)):
                raise ValueError(f"Initial values {x0} must be finite")
        except KeyError as e:
            raise ValueError(f"Parameter ID {e} not found in paramset") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid initial values {x0}: {str(e)}") from e


        # Prepare bounds
        bounds_list = None
        if self.bounds is not None:
            bounds_list = [self.bounds.get(id, (None, None)) for id in self.opt_param_ids]

        # Define supported minimize methods
        minimize_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 
                          'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 
                          'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
        global_methods = ['dual_annealing', 'basinhopping']

        # Run optimization
        try:
            if isinstance(self.optimizer, str):
                if self.optimizer in minimize_methods:
                    results = minimize(objective_wrapper, x0, method=self.optimizer,
                                       bounds=bounds_list, tol=self.tol, options=self.options, **self.kwargs)
                elif self.optimizer in global_methods:
                    if bounds_list is None or any(b[0] is None or b[1] is None or
                                                  not (np.isfinite(b[0]) and np.isfinite(b[1]))
                                                  for b in bounds_list):
                        raise ValueError(f"Finite bounds are required for {self.optimizer}")
                    if self.optimizer == 'dual_annealing':
                        results = dual_annealing(objective_wrapper, bounds_list,
                                                 **{**self.options, **self.kwargs})
                    else:  # basinhopping
                        results = basinhopping(objective_wrapper, x0, 
                                               **{**self.options, **self.kwargs})
                else:
                    raise ValueError(f"Unsupported optimizer string: {self.optimizer}")
                if not isinstance(results, OptimizeResult):
                    raise ValueError(f"'{self.optimizer}' optimizer did not return a scipy.optimize.OptimizeResult object")
            else:  # Callable optimizer
                try:
                    results = self.optimizer(objective_wrapper, x0, bounds=bounds_list,
                                             options=self.options, **self.kwargs)
                    if not isinstance(results, OptimizeResult):
                        raise ValueError("Custom optimizer must return a scipy.optimize.OptimizeResult object")
                    required_attrs = ['x', 'fun', 'success', 'message']
                    if not all(hasattr(results, attr) for attr in required_attrs):
                        raise ValueError(f"Custom optimizer result missing required attributes: {required_attrs}")
                except Exception as e:
                    raise RuntimeError(f"Custom optimizer failed: {str(e)}") from e

            # Validate results
            if not results.success:
                raise RuntimeError(f"Optimization did not converge: {results.message}")
            if len(self.opt_param_ids) != len(results.x):
                raise ValueError(f"Result length {len(results.x)} does not match opt_param_ids length {len(self.opt_param_ids)}")
            if not all(np.isfinite(results.x)):
                raise ValueError(f"Optimization returned non-finite parameters: {results.x}")
            if not np.isfinite(results.fun):
                raise ValueError(f"Optimization returned non-finite objective value: {results.fun}")

            # Update parameter set
            next_paramset.update_param_values(dict(zip(self.opt_param_ids, results.x)))
            obj_value = results.fun if self.direction == 'min' else -results.fun
            next_paramset.update_param(self.assign_to, value=obj_value)

        except Exception as e:
            raise RuntimeError(f"Optimization error: {str(e)}") from e
        
        # Validate optimization results
        try:
            # Check for success
            if not results.success:
                raise RuntimeError(f"Optimization did not converge: {results.message}")

            # Validate result shape
            if len(self.opt_param_ids) != len(results.x):
                raise ValueError(f"Result length {len(results.x)} does not match opt_param_ids length {len(self.opt_param_ids)}")

            # Validate result values
            if not all(np.isfinite(results.x)):
                raise ValueError(f"Optimization returned non-finite parameters: {results.x}")
            if not np.isfinite(results.fun):
                raise ValueError(f"Optimization returned non-finite objective value: {results.fun}")

            # Set optimized parameters
            next_paramset.update_param_values(dict(zip(self.opt_param_ids, results.x)))

            # Set objective value, adjusting for direction
            obj_value = results.fun if self.direction == 'min' else -results.fun
            next_paramset.update_param(self.assign_to, value=obj_value)

        except Exception as e:
            raise RuntimeError(f"Error processing optimization results: {str(e)}") from e

        return (next_paramset,)
    
    # TODO: implement kw optimizer
    # def run(self, paramset: ParameterSet) -> Tuple[ParameterSet, ...]:
    #     """Run the optimization step on the given parameter set."""
    #     # Validate
    #     if self.condition is not None and not self.condition(paramset):
    #         return (paramset,)

    #     # Copy
    #     next_paramset = copy.deepcopy(paramset)

    #     # Initial parameters (x0)
    #     try:
    #         x0 = np.array([next_paramset[id].value for id in self.opt_param_ids], dtype=float)
    #         if not all(np.isfinite(x0)):
    #             raise ValueError(f"Initial values {x0} must be finite")
    #     except KeyError as e:
    #         raise ValueError(f"Parameter ID {e} not found in paramset") from e
    #     except (TypeError, ValueError) as e:
    #         raise ValueError(f"Invalid initial values {x0}: {str(e)}") from e

    #     # Prepare bounds
    #     bounds_list = None
    #     if self.bounds is not None:
    #         bounds_list = [self.bounds.get(id, (None, None)) for id in self.opt_param_ids]

    #     # Run optimization using minimize_keywords
    #     try:
    #         results = minimize_keywords(
    #             func=self.func,
    #             x0=x0,
    #             param_ids=self.opt_param_ids,
    #             method=self.optimizer,
    #             bounds=bounds_list,
    #             constraints=self.kwargs.get('constraints', ()),
    #             tol=self.tol,
    #             options=self.options,
    #             direction=self.direction,
    #             **{k: v for k, v in self.kwargs.items() if k != 'constraints'}
    #         )

    #         # Validate results
    #         if not results.success:
    #             raise RuntimeError(f"Optimization did not converge: {results.message}")
    #         if len(self.opt_param_ids) != len(results.x):
    #             raise ValueError(f"Result length {len(results.x)} does not match opt_param_ids length {len(self.opt_param_ids)}")
    #         if not all(np.isfinite(results.x)):
    #             raise ValueError(f"Optimization returned non-finite parameters: {results.x}")
    #         if not np.isfinite(results.fun):
    #             raise ValueError(f"Optimization returned non-finite objective value: {results.fun}")

    #         # Update parameter set
    #         next_paramset.update_param_values(dict(zip(self.opt_param_ids, results.x)))
    #         next_paramset.update_param(self.assign_to, value=results.fun)

    #     except Exception as e:
    #         raise RuntimeError(f"Optimization error: {str(e)}") from e

    #     return (next_paramset,)
    
    def get_details(self) -> Dict[str, Any]:
        details = super().get_details()
        details.update({
            'func': getattr(self.func, '__name__', '<lambda>'),
            'assign_to': self.assign_to,
            'opt_param_ids': self.opt_param_ids,
            'optimizer': getattr(self.optimizer, '__name__', self.optimizer) if callable(self.optimizer) else self.optimizer,
            'direction': self.direction,
            'tol': self.tol,
            'bounds': self.bounds,
            'options': self.options
        })
        return details

# class End(AnalysisStep):
#     def __init__(self, 
#             # save: bool = True, 
#             condition: Optional[Callable] = None
#         ):
#         super().__init__(condition)

#     def run(self, paramset):
#         if self.condition:
#             return None
#         else:
#             return super().run(paramset)
        
#     def get_details(self) -> Dict[str, Any]:
#         details = super().get_details()
#         details.update({
#             'condition': getattr(self.condition, '__name__', '<lambda>') if self.condition else None
#         })
#         return details