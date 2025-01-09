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


from typing import Callable
import logging
import copy
import pandas as pd
from datetime import datetime
from time import time
import numpy as np
from scipy.optimize import minimize

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
            input_param_ids: str = None,
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

        self.input_param_ids = input_param_ids

    def run(self, paramset):
        """
        Executes function and modifies paramset

        TODO future args: auto_add_new_args=True
        """
        super().run(paramset)

        # TODO might need to deepcopy params is preservation of the upper level is needed
        next_paramset = copy.deepcopy(paramset)

        # Execute function: Positional argument input mode
        if self.input_param_ids:
            output = next_paramset.call_with_positional_args(func=self.func, param_ids=self.input_param_ids)
        
        # Execute function: Keyword argument input mode
        else: 
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
    """
    def __init__(self,
            func,
            obj_param_id = None,
            opt_param_ids = None,
            method = 'SLSQP',
            ftol = None,
            xtol = None,
            func_output_mode = 'single',
            condition = None
        ):
        super().__init__(condition)

        # Validate stuff
        # if not is_valid_kwarg_func(func): TODO make this function validator a thing
        #     raise ValueError(f"Invalid Optimize: 'func' is not a valid objective function: {obj_param_id}")
        if not isinstance(func, Callable):
            raise ValueError(f"Invalid Optimize: 'func' is not a valid objective function: {obj_param_id}")        
        self.func = func

        if not Parameter.is_valid_id(obj_param_id):
            raise ValueError(f"Invalid Optimize: 'obj_param_id' is not a valid param id: {obj_param_id}")
        self.obj_param_id = obj_param_id

        # TODO functionalize these checks so they can be one-liners
        if not Parameter.is_valid_id_iterable(opt_param_ids):
            raise ValueError(f"Invalid Optimize: 'obj_param_id' is not a valid param id tuple: {opt_param_ids}")        
        self.opt_param_ids = opt_param_ids

        self.method = method
        self.ftol = ftol
        self.xtol = xtol
        self.func_output_mode = func_output_mode
        
    def run(self, paramset):
        # Validate
        super().run(paramset)

        # Copy
        next_paramset = copy.deepcopy(paramset)

        # create objective wrapper where x is a list of opt vars
        def objective_wrapper(x): # (func, next_paramset, opt_param_ids)
            '''
            evaluates the user supplied objective function using fresh optimizer values from x.
            uses x to modify parameter set and then run user objective function
            '''
            # Create test parameter set with values from optimizer
            test_paramset = copy.deepcopy(next_paramset)
            test_paramset.update_param_values(dict(zip(self.opt_param_ids, x)))

            # evaluate objective parameter
            obj_val = test_paramset.call_with_matched_kwargs(self.func)

            # TODO add code for different output methods

            return obj_val
        
        # Create x0 array
        x0 = [next_paramset[id].value for id in self.opt_param_ids]
        
        # run optimization
        results = minimize(objective_wrapper, x0, method=self.method)

        # Check if the number of optimization parameters matches the length of x
        if len(self.opt_param_ids) != len(results.x):
            raise ValueError("Number of optimization parameters does not match the length of the optimization result")
        
        # Set optimization parameters
        next_paramset.update_param_values(dict(zip(self.opt_param_ids, results.x)))

        # Set objective parameter
        next_paramset.update_param(self.obj_param_id, value=results.fun)

        return (next_paramset,)

class End(AnalysisStep):
    def __init__(self, condition = None):
        super().__init__(condition)

    def run(self, paramset):
        if self.condition:
            return None
        else:
            return super().run(paramset)

class AnalysisModule():
    def __init__(self, name='') -> None:

        self.sequence = []
        self.finished_leaves:int = 0

        self.name = name

        self.df: pd.DataFrame = pd.DataFrame()

        self.t0 = None

        pass

    def add(self, analysis_object: AnalysisStep) -> None:
        '''
        appends AnalysisStep to the Analysis Sequence
        e.g.: define, fork, evaluation or optimization
        '''
        # TODO Validate AnalysisStep

        # TODO analysis_object.initialize(self.sequence)

        self.sequence.append(analysis_object)

    def run(self,
            create_pandas = True,
            verbose = True
        ):
        # self.sequence = tuple(self.sequence)

        self.t0 = time()
        self.finished_leaves = 0
        self.execute_step(ParameterSet(), step_idx=0) # start on step 0

        print('done with analysis!')

        results = dict(
            df = self.df,
            runtime = self.runtime,
            finished_leaves = self.finished_leaves,
        )

        return results

    def execute_step(self, paramset: ParameterSet, step_idx:int):
        """
        Runs a single step of the analysis sequence, then recursively calls the next.
        """

        # End analysis branch if no more steps
        if step_idx >= len(self.sequence):
            self.end_sequence(paramset)
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
            # Add this steps runtime to total runtime
            ps['runtime'].value += step_runtime

            # Recursivly call next step
            self.execute_step(ps, step_idx+1)

        # End analysis branch if no paramset returned
        # TODO decide how to handle End: (paramset is None) OR (type(analysis_step)==End)
        if new_paramsets is None:
            self.end_sequence(paramset)
        
        return
    
    def end_sequence(self, paramset: ParameterSet):
        # increment leaf count
        self.finished_leaves += 1

        # Add a new row to DataFrame
        new_row = paramset.values_dict.copy()  # Copy the values_dict
        new_row['datetime'] = datetime.now()  # Add the current datetime

        # Append the new row
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

        self.runtime = time() - self.t0
