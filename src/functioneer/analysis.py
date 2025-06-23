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


from typing import Callable, Dict
import logging
import copy
import pandas as pd
from datetime import datetime
from time import time
import numpy as np
from scipy.optimize import minimize, dual_annealing

from functioneer.parameter import ParameterSet, Parameter
from functioneer.util import call_with_matched_kwargs

class AnalysisStep():
    """
    Template AnalysisStep object containing basic info
    """
    def __init__(self, condition: Callable[..., bool] | None = None) -> None:
        if condition is not None and not isinstance(condition, Callable):
            raise ValueError(f"AnalysisStep condition must be a function that returns a bool")
        
        self.condition = condition

        # self.branch_cnt = 1

    def run(self, paramset: ParameterSet):
        """
        template method for executing the AnalysisStep
        returns a list of one or more ParameterSets that have been altered according to the AnalysisStep
        returns a tuple of one or more parametersets

        TODO decide if Default behavior is to COPY or PASS the paramset

        validates paramset
        """
        if not isinstance(paramset, ParameterSet):
            # logging.error(f"paramset is not of type ParameterSet")
            raise ValueError(f"paramset is not of type ParameterSet")

        return (paramset,)
    
class Define(AnalysisStep):
    """
    Define AnalysisStep: Adds parameter to parameterset
    Will create new Parameter in ParameterSet if one does not already exist.
    """
    def __init__(self, 
            name: str,
            value = None,
            # func = None,
            # func_output_mode = 'single',
            condition = None
        ):
        super().__init__(condition)

        if isinstance(name, str):
            self.parameter = Parameter(name, value)

        # elif isinstance(param, Parameter):
        #     pass

        else:
            logging.error(f"param must be of type Parameter")

    def run(self, paramset):
        # Validate
        super().run(paramset)

        # add new Parameter to Paramset
        paramset.add_param(self.parameter)
        
        return (paramset,)
    
class Fork(AnalysisStep):
    """
    Fork AnalysisStep: Splits analysis into several parallel analysis based on the provided parameters and values.
    Will create new Parameter in ParameterSet if one does not already exist.
    """
    def __init__(self, 
            param_ids,
            value_sets, # value_sets
            condition = None
            ):
        super().__init__(condition)

        # Handle single parameter id
        if Parameter.is_valid_id(param_ids):
            self.param_ids = (param_ids,)
            value_sets = (value_sets,)
            
        # Handle multiple param ids
        elif isinstance(param_ids, (tuple, list)) and all([Parameter.is_valid_id(id) for id in param_ids]):
            self.param_ids = param_ids
            
        else:
            raise ValueError(f"Fork param_ids must be a valid param id or a tuple of valid param ids")
        
        # Identify number of parameters
        self.param_cnt = len(self.param_ids)

        # Validate same number of params and value sets
        if len(self.param_ids) != len(value_sets):
            raise ValueError(f"Invalid Fork (param_ids='{self.param_ids}'): Mismatch between number of parameters and number or value sets.")
        
        self.value_cnt = len(value_sets[0])
        for vs in value_sets:
            if len(vs) != self.value_cnt:
                raise ValueError(f"Invalid Fork (param_ids='{self.param_ids}'): Each parameter's value set must be the same length")

        self.value_sets = value_sets

        # TODO Validate value types
        # this will be optional but left alone and values are tuples then it might be good to throw a warning in case user meant 
        # raise ValueError(f"Warning: values of paramfork are of type 'tuple'. to silence this error set Parameter.value_type tot tuple")

    def run(self, paramset: dict[str, Parameter]):
        super().run(paramset)
        
        # Do forky stuff
        
        next_paramsets = []
        for branch_idx in range(self.value_cnt):
            ps: ParameterSet = copy.deepcopy(paramset)
            for id, vs in zip(self.param_ids, self.value_sets):
                val = vs[branch_idx]
                ps.update_param(id, val)

            next_paramsets.append(ps)

        return next_paramsets
    
class Execute(AnalysisStep):
    """
    Execute AnalysisStep: Executes provided function using matched kwargs and sets parameters based on returned dict
    """
    def __init__(self,
            func: Callable,
            output_param_ids: str = None,
            # input_param_ids: str[] = None, ## allows for mapping to positional args (TODO: maybe this should move to a function wrapper object)
            condition = None
        ):
        super().__init__(condition)

        self.func = func

        if output_param_ids is None:
            self.output_param_ids = output_param_ids
        elif Parameter.is_valid_id(output_param_ids):
            self.output_param_ids = output_param_ids
        elif Parameter.is_valid_id_iterable(output_param_ids):
            self.output_param_ids = output_param_ids
        else:
            raise ValueError(f"Execute Step's output_param_ids must be a str or tuple of str")

        # self.input_param_ids = input_param_ids

    def run(self, paramset):
        """
        Executes function and modifies paramset

        TODO future args: auto_add_new_args=True
        """
        super().run(paramset)

        # TODO might need to deepcopy params is preservation of the upper level is needed
        next_paramset = copy.deepcopy(paramset)

        # Execute function: Positional argument input mode
        # if self.input_param_ids:
        #     output = next_paramset.call_with_positional_args(func=self.func, param_ids=self.input_param_ids)
        
        # Execute function: Keyword argument input mode
        # else: 
        output = next_paramset.call_with_matched_kwargs(func=self.func)

        # Process output: Direct output mode
        if self.output_param_ids and isinstance(self.output_param_ids, str):
            next_paramset.update_param(id=self.output_param_ids, value=output)

        # Process output: Positional output mode
        elif self.output_param_ids:
            if len(self.output_param_ids) != len(output):
                raise ValueError(f"Number of function outputs ({len(output)}) does not match the specified output_param_ids ({len(self.output_param_ids)})")
            
            for i, id in enumerate(self.output_param_ids):
                next_paramset.update_param(id=id, value=output[i])

        # Process output: Keyword output mode
        else:
            if not isinstance(output, dict):
                raise ValueError(f"function in step {69} does not return valid dict of param names and values")
                        
            # next_paramsets.update_values(results)

            # TODO this should be functionalized
            # paramset.update_param_values(results)
            for id, val in output.items():
                next_paramset.update_param(id, value=val)

        return (next_paramset,)

class Optimize(AnalysisStep):
    """
    Optimize AnalysisStep: Minimizes (or maximizes) objective by optimizing opt vars

    Parameters
    ----------
    func : callable
        The objective function to optimize. Must return a scalar value.
    assign_to : str, optional
        Parameter ID where the optimized objective value is stored.
    opt_param_ids : iterable of str, optional
        Parameter IDs to optimize.
    direction : {'min', 'max'}, optional
        Direction of optimization. Default is 'min' (minimization).
    optimizer : str, optional
        Optimization method to use. Default is 'SLSQP'. Options include:
        - SciPy minimize methods: 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
          'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', etc.
        - Simulated annealing: 'dual_annealing' (requires finite bounds).
    bounds : dict, optional
        Dictionary mapping opt_param_ids to (min, max) tuples. E.g., {'x': (0, 1)}.
        Required for 'dual_annealing', optional otherwise.
    options : dict, optional
        Additional options for the optimizer. Common options:
        - 'ftol': float, precision goal for the objective function value.
        - 'xtol': float, precision goal for the parameter values.
        - 'maxiter': int, maximum number of iterations.
        - 'disp': bool, whether to print convergence messages.
        See SciPy's minimize or dual_annealing documentation for more.
    condition : callable, optional
        Condition function for the analysis step.
    """
    def __init__(self,
            func,
            assign_to = None,
            opt_param_ids = None,
            direction='min',
            optimizer='SLSQP',
            bounds=None,
            options = None,
            condition = None
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
            else:
                assign_to = func.__name__
        
        # Validate assign_to
        if not Parameter.is_valid_id(assign_to):
            raise ValueError(f"Invalid Optimize: 'assign_to' is not a valid param id: {assign_to}")

        self.assign_to = assign_to

        # Validate opt_param_ids
        if not Parameter.is_valid_id_iterable(opt_param_ids):
            raise ValueError(f"Invalid Optimize: 'opt_param_ids' is not a valid param id tuple: {opt_param_ids}")
        self.opt_param_ids = opt_param_ids

        # Validate direction
        if direction not in ['min', 'max']:
            raise ValueError(f"Invalid Optimize: 'direction' must be 'min' or 'max', got {direction}")
        self.direction = direction

        # Validate optimizer
        if not isinstance(optimizer, str):
            raise ValueError(f"Invalid Optimize: 'optimizer' must be a string, got {type(optimizer)}")
        self.optimizer = optimizer

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
        
    def run(self, paramset):
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
                obj_val = test_paramset.call_with_matched_kwargs(self.func) # evaluate objective parameter
                if not isinstance(obj_val, (int, float, np.number)):
                    raise ValueError(f"Objective function returned non-scalar value: {obj_val}")
                if not np.isfinite(obj_val):
                    raise ValueError(f"Objective function returned non-finite value: {obj_val}")
                return obj_val if self.direction == 'min' else -obj_val
            except Exception as e:
                raise RuntimeError(f"Error evaluating objective function: {str(e)}") from e
        
        # Create x0 array
        try:
            x0 = [next_paramset[id].value for id in self.opt_param_ids]
            if not all(isinstance(v, (int, float, np.number)) for v in x0):
                raise ValueError(f"Initial values {x0} must be numeric")
            if not all(np.isfinite(v) for v in x0):
                raise ValueError(f"Initial values {x0} must be finite")
        except KeyError as e:
            raise ValueError(f"Parameter ID {e} not found in paramset") from e


        # Prepare bounds
        if self.bounds is not None:
            bounds_list = [self.bounds.get(id, (None, None)) for id in self.opt_param_ids]
        else:
            bounds_list = None

        # Define supported minimize methods
        minimize_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 
                          'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 
                          'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
        
        # Run optimization
        try:
            if self.optimizer in minimize_methods:
                results = minimize(objective_wrapper, x0, method=self.optimizer,
                                   bounds=bounds_list, options=self.options)
            elif self.optimizer == 'dual_annealing':
                if bounds_list is None or any(b[0] is None or b[1] is None or
                                              not (np.isfinite(b[0]) and np.isfinite(b[1]))
                                              for b in bounds_list):
                    raise ValueError("Finite bounds are required for dual_annealing")
                results = dual_annealing(objective_wrapper, bounds_list, **self.options)
            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        except Exception as e:
            raise RuntimeError(f"Optimization failed with {self.optimizer}: {str(e)}") from e

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

class End(AnalysisStep):
    def __init__(self, condition = None):
        super().__init__(condition)

    def run(self, paramset):
        if self.condition:
            return None
        else:
            return super().run(paramset)

## TODO: work towards staged analysis (needed for pick best 10) 
# allow for a tuple of dicts for starting with multiple parameter sets, 
# also might allow sending in of the pandas datafram to add to it (if not just append new rows when done with sub analysis)
# pandas_to_paramsets: a function that takes in a pd and returns a tuple of paramsets ready to be fed into the next stage of analysis
class AnalysisModule():
    def __init__(self, init_param_values={}, name='') -> None:

        # Analysis Setup
        self.sequence: list[AnalysisStep] = []
        self.finished_leaves:int = 0
        self.init_paramset: ParameterSet = ParameterSet()
        self.init_paramset.update_param_values(init_param_values)

        # Namespaces
        self.add = self.AddNamespace(self)  # Instantiate the namespace

        # Results and Metadata
        self.name = name
        self.df: pd.DataFrame = pd.DataFrame()
        self.t0 = None

        pass

    class AddNamespace:
        """Namespace for adding different types of analysis steps."""

        def __init__(self, parent):
            self.parent: AnalysisModule = parent
            self.define = self.DefineNamespace(parent)
            self.fork = self.ForkNamespace(parent)
            self.optimize = self.OptimizeNamespace(parent)
            self.execute = self.ExecuteNamespace(parent)

        def __call__(self,
            analysis_object: AnalysisStep
        ) -> None:
            '''
            appends AnalysisStep to the Analysis Sequence
            e.g.: define, fork, evaluation or optimization
            '''
            # TODO Validate AnalysisStep

            # TODO analysis_object.initialize(self.sequence)

            self.parent.sequence.append(analysis_object)

        class DefineNamespace:
            def __init__(self, parent):
                self.parent: AnalysisModule = parent

            def __call__(self, 
                name: str,
                value = None,
                # func = None,
                # func_output_mode = 'single',
                condition = None
            ):
                self.parent.sequence.append(Define(name, value, condition))

        class ForkNamespace:
            def __init__(self, parent):
                self.parent: AnalysisModule = parent

            def __call__(self, 
                param_id,
                value_set, # value_sets
                condition = None
            ):
                self.parent.sequence.append(Fork(param_id, value_set, condition))

            def multi(self,
                param_ids: tuple[str, ...],
                value_sets: tuple[tuple, ...],
                condition: Callable[[], bool] = None
            ):
                self.parent.sequence.append(Fork(param_ids, value_sets, condition))

            # def copy(self,
            #     n: int,
            #     condition: Callable[[], bool] = None
            # ):
            #     raise TypeError("Fork.copy not setup yet")
            #     self.parent.sequence.append(Fork(n, condition))

            
        class ExecuteNamespace:
            def __init__(self, parent):
                self.parent: AnalysisModule = parent

            def __call__(self, 
                func: Callable,
                output_param_ids: str = None,
                # input_param_ids: str = None,
                condition = None
            ):
                # self.parent.sequence.append(Execute(func, output_param_ids, input_param_ids, condition))
                self.parent.sequence.append(Execute(func, output_param_ids, condition))

            def dict_return(self,
                func: Callable[..., Dict[str, float]],
                condition = None
            ):
                self.parent.sequence.append(Execute(func, ))

        class OptimizeNamespace:
            def __init__(self, parent):
                self.parent: AnalysisModule = parent

            def __call__(self, 
                func: Callable,
                assign_to: str = None,
                opt_param_ids: str = None,
                direction='min',
                optimizer='SLSQP',
                bounds=None,
                options = None,
                condition = None
            ):
                """
                Optimize AnalysisStep: Minimizes (or maximizes) objective by optimizing opt vars

                Parameters
                ----------
                func : callable
                    The objective function to optimize. Must return a scalar value.
                assign_to : str, optional
                    Parameter ID where the optimized objective value is stored.
                opt_param_ids : iterable of str, optional
                    Parameter IDs to optimize.
                direction : {'min', 'max'}, optional
                    Direction of optimization. Default is 'min' (minimization).
                optimizer : str, optional
                    Optimization method to use. Default is 'SLSQP'. Options include:
                    - SciPy minimize methods: 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                    'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', etc.
                    - Simulated annealing: 'dual_annealing' (requires finite bounds).
                bounds : dict, optional
                    Dictionary mapping opt_param_ids to (min, max) tuples. E.g., {'x': (0, 1)}.
                    Required for 'dual_annealing', optional otherwise.
                options : dict, optional
                    Additional options for the optimizer. Common options:
                    - 'ftol': float, precision goal for the objective function value.
                    - 'xtol': float, precision goal for the parameter values.
                    - 'maxiter': int, maximum number of iterations.
                    - 'disp': bool, whether to print convergence messages.
                    See SciPy's minimize or dual_annealing documentation for more.
                condition : callable, optional
                    Condition function for the analysis step.
                """
                self.parent.sequence.append(Optimize(
                    func,
                    assign_to,
                    opt_param_ids,
                    direction,
                    optimizer,
                    bounds,
                    options,
                    condition
                ))


    def run(self,
            # create_pandas = True,
            # verbose = True
        ):
        # self.sequence = tuple(self.sequence)

        self.t0 = time()
        self.finished_leaves = 0
        self._execute_step(self.init_paramset, step_idx=0) # start on step 0

        print('done with analysis!')

        results = dict(
            df = self.df,
            runtime = self.runtime,
            finished_leaves = self.finished_leaves,
        )

        return results

    def _execute_step(self, paramset: ParameterSet, step_idx:int):
        """
        Runs a single step of the analysis sequence, then recursively calls the next.
        """

        # End analysis branch if no more steps
        if step_idx >= len(self.sequence):
            self._end_sequence(paramset)
            return

        # Extract step object
        analysis_step: AnalysisStep = self.sequence[step_idx]

        # Check if step has a conditional
        run_step = True
        if analysis_step.condition is not None:
            run_step = paramset.call_with_matched_kwargs(analysis_step.condition)

        t0 = time()
        if run_step:
            # Run the next step
            new_paramsets = analysis_step.run(paramset)
        else: # pass step
            new_paramsets = (copy.deepcopy(paramset),)
        step_runtime = time() - t0

        # Process Step and Recursivly call next step
        for ps in new_paramsets:
            # Add this steps runtime to total row runtime
            ps['runtime'].value += step_runtime

            # Recursivly call next step
            self._execute_step(ps, step_idx+1)

        # End analysis branch if no paramset returned
        # TODO decide how to handle End: (paramset is None) OR (type(analysis_step)==End)
        if new_paramsets is None:
            self._end_sequence(paramset)
        
        return
    
    def _end_sequence(self, paramset: ParameterSet):
        # increment leaf count
        self.finished_leaves += 1

        # Add a new row to DataFrame
        new_row = paramset.values_dict.copy()  # Copy the values_dict
        new_row['datetime'] = datetime.now()  # Add the current datetime

        # Append the new row
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

        self.runtime = time() - self.t0
