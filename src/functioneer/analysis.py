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
import pandas as pd
from datetime import datetime
import time
from tqdm.notebook import tqdm

from functioneer.steps import AnalysisStep, Define, Fork, Evaluate, Optimize
from functioneer.parameter import ParameterSet, Parameter

## TODO: work towards staged analysis (eg. needed for pick best 10) 
# allow for a tuple of dicts for starting with multiple parameter sets, 
# also might allow sending in of the pandas datafram to add to it (if not just append new rows when done with sub analysis) 
# pandas_to_paramsets: a function that takes in a pd and returns a tuple of paramsets ready to be fed into the next stage of analysis 
class AnalysisModule():
    """
    The central container for an analysis pipeline in functioneer.

    Parameters
    ----------
    init_param_values : dict, optional
        Initial parameter values as a dictionary with parameter IDs (str) as keys and their values.
        Parameter IDs must be valid (non-empty strings, not reserved names like 'runtime' or 'datetime').
        Defaults to an empty dictionary.
    name : str, optional
        Name of the analysis module. Defaults to an empty string.

    Raises
    ------
    ValueError
        If init_param_values is not a dictionary, contains invalid parameter IDs, or includes reserved names.
    TypeError
        If name is not a string.
    """
    def __init__(self, init_param_values={}, name='', error_handling='skip_leaf') -> None:
        """ Initialize Functioneer Analysis

        Args:
            init_param_values : dict, optional
                Initial parameter values as a dictionary with parameter IDs (str) as keys and their values.
                Parameter IDs must be valid (non-empty strings, not reserved names like 'runtime' or 'datetime').
                Defaults to an empty dictionary.
            name : str, optional
                Name of the analysis module. Defaults to an empty string.
            error_handling : str, optional
                Determines behavior on errors in steps. Options: 'exit' (raises error and stops analysis),
                'skip_leaf' (skips the current branch and records partial data), 'continue_leaf' (continues
                the branch, skipping the failed step). Defaults to 'skip_leaf'.

        Raises:
            ValueError: If init_param_values is not a dictionary, contains invalid parameter IDs, or includes reserved names.
            If error_handling is not one of the valid options.
            TypeError: If name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError(f"Invalid AnalysisModule: 'name' must be a string, got {type(name)}")
        if not isinstance(init_param_values, dict):
            raise ValueError(f"Invalid AnalysisModule: 'init_param_values' must be a dictionary, got {type(init_param_values)}")
        for param_id in init_param_values:
            Parameter.validate_id(param_id)

        valid_error_handling = ['exit', 'skip_leaf', 'continue_leaf']
        if error_handling not in valid_error_handling:
            raise ValueError(f"Invalid error_handling: must be one of {valid_error_handling}, got '{error_handling}'")

        self.name = name
        self.sequence: list[AnalysisStep] = []
        self.init_paramset: ParameterSet = ParameterSet()
        self.init_paramset.update_param_values(init_param_values)
        self.init_paramset.add_param(Parameter('runtime', 0))
        self.finished_leaves: int = 0
        self.total_leaves: int = 0  # Total leaves for progress tracking
        self.progress_bar: Optional[tqdm] = None
        self.show_notebook_progress_bar: bool = False
        self.error_handling = error_handling
        self.errors: List[Dict[str, Any]] = []

        # Namespaces
        self.add = self.AddNamespace(self)  # Instantiate the namespace

        # Results and Metadata
        self.df: pd.DataFrame = pd.DataFrame()
        self.t0 = None
        self.leaf_data: List[Dict[str, Any]] = []
        self.runtime: Optional[float] = None

        pass

    class AddNamespace:
        """Namespace for adding different types of analysis steps."""
        def __init__(self, parent):
            self.parent: AnalysisModule = parent

        def __call__(self,
            analysis_object: AnalysisStep
        ) -> None:
            """Manually append an AnalysisStep object to the Analysis
            """
            # TODO Validate AnalysisStep

            # TODO analysis_object.initialize(self.sequence)

            self.parent.sequence.append(analysis_object)

        def define(self, param_or_dict: Union[str, Dict[str, Any]], value: Any = None, condition: Optional[Callable[..., bool]] = None) -> None:
            """Define a parameter with a single ID and value or a dictionary of parameters.

            Args:
                param_or_dict: Parameter ID (str) or dictionary of parameter IDs to values.
                value: Value for the parameter (if param_or_dict is a string).
                condition: Optional condition function to determine if the step should run.

            Examples:
                >>> anal.add.define('a', 1)  # Define single parameter
                >>> anal.add.define({'a': 1, 'b': 100})  # Define multiple parameters

            Raises:
                ValueError: If inputs are invalid (e.g., wrong types, missing value, or value provided with dict).
            """
            if isinstance(param_or_dict, dict):
                if value is not None:
                    raise ValueError(
                        "When defining multiple parameters with a dictionary, the 'value' argument is ignored. "
                        "Use either define(param_id: str, value: Any) or define(params: Dict[str, Any])."
                    )
                for param_id, val in param_or_dict.items():
                    self.parent.sequence.append(Define(param_id, val, condition))
            elif isinstance(param_or_dict, str) and value is not None:
                self.parent.sequence.append(Define(param_or_dict, value, condition))
            else:
                raise ValueError("Expected (param_id, value) or {param_id: value, ...}")
            
        def fork(self, 
            param_or_dict_or_configs: Union[str, Dict[str, Union[Tuple[Any, ...], np.ndarray]], List[Dict[str, Any]], Tuple[Dict[str, Any], ...]], 
            value_list: Optional[Union[Tuple[Any, ...], np.ndarray]] = None, 
            condition: Optional[Callable[..., bool]] = None
        ) -> None:
            """Fork analysis with a single parameter, dictionary of parameter value lists, or list of parameter configurations.

            Args:
                param_or_dict_or_configs: Parameter ID (str), dictionary of parameter IDs to value lists, or list/tuple of parameter configurations.
                value_list: Tuple, List or 1-D np.array of values for the parameter (if param_or_dict_or_configs is a string).
                condition: Optional condition function to determine if the step should run.

            Examples:
                >>> anal.add.fork('x', (0, 1, 2))  # Fork single parameter
                >>> anal.add.fork('x', np.array([0, 1, 2]))  # Single parameter with numpy array
                >>> anal.add.fork({'x': (0, 1), 'y': (10, 20)})  # Fork multiple parameters
                >>> anal.add.fork({'x': np.array([0, 1]), 'y': np.array([10, 20])})  # Multiple parameters with numpy arrays
                >>> anal.add.fork([{'x': 0, 'y': 0}, {'x': 1, 'y': 10}])  # Fork with parameter configurations

            Raises:
                ValueError: If inputs are invalid (e.g., wrong types, missing value_list, non-iterable value lists, non-1D numpy arrays, or inconsistent configurations).
            """
            # Single Parameter
            if isinstance(param_or_dict_or_configs, str):
                if value_list is None:
                    raise ValueError("value_list must be provided for single parameter fork")
                if isinstance(value_list, np.ndarray):
                    if value_list.ndim != 1:
                        raise ValueError("value_list must be 1D numpy array for single parameter")
                    value_list = value_list.tolist()  # Convert numpy array to list for consistency
                if not isinstance(value_list, (list, tuple)):
                    raise ValueError("value_list must be a list or tuple of parameter values")
                configurations = [{param_or_dict_or_configs: value} for value in value_list]
                # raise DeprecationWarning(".fork('param', (0, 1, 2)) is being depreciated, please use a 'value dict' .fork({'param', (0, 1, 2)})")

            # Multiple Parameters
            elif isinstance(param_or_dict_or_configs, dict):
                param_ids = list(param_or_dict_or_configs.keys())
                value_lists_raw = list(param_or_dict_or_configs.values())
                value_lists = []
                # Validate and convert value lists
                for vs in value_lists_raw:
                    if isinstance(vs, np.ndarray):
                        if vs.ndim != 1:
                            raise ValueError("Value arrays in dictionary must be 1D")
                        value_lists.append(vs.tolist())  # Convert numpy array to list
                    elif isinstance(vs, (list, tuple)):
                        value_lists.append(vs)
                    else:
                        raise ValueError("Values in dictionary must be lists, tuples, or 1D numpy arrays")
                    
                # Validate that value lists are non-empty and have the same length
                value_lengths = [len(values) for values in value_lists]
                if not value_lengths:
                    raise ValueError("param_value_lists dictionary cannot be empty")
                if len(set(value_lengths)) > 1:
                    raise ValueError("All value lists must have the same length")
                
                # Create configurations
                value_configs = zip(*value_lists)
                configurations = [dict(zip(param_ids, values)) for values in value_configs]

            # Parameter Configurations
            elif isinstance(param_or_dict_or_configs, (list, tuple)):
                configurations = list(param_or_dict_or_configs) # Convert to list for consistency
                # Validate that configurations is a list of dictionaries
                if not all(isinstance(config, dict) for config in configurations):
                    raise ValueError("Configurations must be a list or tuple of dictionaries")

            else:
                raise ValueError("Invalid input for fork: expected str, dict, or list/tuple of dicts")
            
            # Validate parameter IDs in configurations
            for config in configurations:
                for param_id in config:
                    try:
                        Parameter.validate_id(param_id)
                    except ValueError as e:
                        raise ValueError(f"Invalid param_id '{param_id}' in fork: {str(e)}") from e

            self.parent.sequence.append(Fork(configurations, condition))
            
        def evaluate(self, func: Callable[..., Any], assign_to: Optional[Union[str, List[str], Tuple[str, ...]]] = None, unpack_result: bool = False, condition: Optional[Callable[..., bool]] = None) -> None:
            """Evaluate a function and store its result in the ParameterSet.

            Args:
                func: Function to evaluate, taking parameter values as input.
                assign_to: Custom Parameter ID(s) to store the result (str or list/tuple of strings).
                unpack_result: If True, unpacks a dictionary result into multiple parameters.
                condition: Optional condition function to determine if the step should run.

            Examples:
                >>> anal.add.evaluate(my_function)  # Evaluate function, store result
                >>> anal.add.evaluate(my_function, assign_to='new_param')  # Evaluate function, store result to param: 'new_param'
                >>> anal.add.evaluate(my_function_returns_dict, unpack_result=True)  # Unpack dict result
                >>> anal.add.evaluate(my_function_returns_dict, assign_to=['out_1', 'out_2'], unpack_result=True)  # Unpack only dict keys: out_1, out_2

            Raises:
                ValueError: If func is not callable or assign_to is invalid.
            """
            self.parent.sequence.append(Evaluate(func, assign_to, unpack_result, condition))
                
        def optimize(self, func: Callable[..., float], opt_param_ids: Tuple[str, ...], assign_to: Optional[str] = None, direction: str = 'min', optimizer: Union[str, Callable] = 'SLSQP', tol: Optional[float] = None, bounds: Optional[Dict[str, Tuple[float, float]]] = None, options: Optional[Dict[str, Any]] = None, condition: Optional[Callable[..., bool]] = None, **kwargs) -> None:
            """Optimize a function over specified parameters.

            Args:
                func: Objective function to optimize, returning a scalar.
                opt_param_ids: Parameter IDs to optimize.
                assign_to: Parameter ID to store the optimized value.
                direction: Optimization direction ('min' or 'max').
                optimizer: Optimization method (SciPy method name or callable).
                tol: Tolerance for convergence.
                bounds: Dictionary of parameter IDs to (min, max) tuples.
                options: Additional optimizer options (e.g., maxiter, ftol).
                condition: Optional condition function to determine if the step should run.
                **kwargs: Additional arguments for the optimizer.

            Examples:
                >>> anal.add.optimize(rosenbrock, ('x', 'y'))  # Minimize with SLSQP
                >>> anal.add.optimize(rosenbrock_neg, ('x', 'y'), direction='max', optimizer='Nelder-Mead')  # Maximize with Nelder-Mead

            Raises:
                ValueError: If inputs are invalid (e.g., invalid direction, optimizer).
            """
            self.parent.sequence.append(Optimize(func, opt_param_ids, assign_to, direction, optimizer, tol, bounds, options, condition, **kwargs))

    def run(self,
            show_notebook_progress_bar = False
            # create_pandas = True,
            # verbose = True
        )  -> Dict[str, Any]:
        """Evaluate the analysis sequence and return results including a DataFrame of leaf data.

        Returns:
            Dict[str, Any]: Dictionary containing the results DataFrame, runtime, and number of finished leaves.
        """
        # Pre analysis
        # Calculate total leaves
        self.total_leaves = 1
        for step in self.sequence:
            if isinstance(step, Fork):
                self.total_leaves *= len(step.configurations)
        # Initialize progress bar
        if show_notebook_progress_bar:
            self.show_notebook_progress_bar = True
            self.progress_bar = tqdm(total=self.total_leaves, desc="Processing leaves", unit="leaf")

        self.t0 = time.time()
        self.finished_leaves = 0
        self.leaf_data = []  # Reset leaf data
        self.errors = []  # Reset errors
        try:
            self._process(self.init_paramset, step_idx=0) # start on step 0
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {str(e)}") from e
        finally:
            self.runtime = time.time() - self.t0

        # Create DataFrame from collected leaf data
        df = pd.DataFrame(self.leaf_data) if self.leaf_data else pd.DataFrame()

        return dict(
            df = df,
            leaf_data = self.leaf_data,
            runtime = self.runtime,
            finished_leaves = self.finished_leaves,
            errors = self.errors,
        )

    def _process(self, paramset: ParameterSet, step_idx:int):
        """
        Run a step of the analysis sequence and recursively process the next step.
        """
        try:
            # Terminate branch if no more steps
            if step_idx >= len(self.sequence):
                self._end_sequence(paramset)
                return

            # Get current step
            step: AnalysisStep = self.sequence[step_idx]

            # Check step condition
            try:
                run_step = paramset.call_with_kwargs(step.condition) if step.condition else True
            except Exception as e:
                error_info = {
                    'step_idx': step_idx,
                    'step_type': type(step).__name__,
                    'details': step.get_details(),
                    'paramset_values': paramset.values_dict.copy(),
                    'error': str(e),
                    'is_condition': True
                }
                self.errors.append(error_info)
                if self.error_handling == 'exit':
                    raise RuntimeError(f"Error evaluating step condition: {str(e)}") from e
                elif self.error_handling == 'skip_leaf':
                    self._end_sequence(paramset)
                    return
                elif self.error_handling == 'continue_leaf':
                    run_step = True

            # Run the analysis step
            t0 = time.time()
            try:
                new_paramsets = step.run(paramset) if run_step else (copy.deepcopy(paramset),)
            except Exception as e:
                error_info = {
                    'step_idx': step_idx,
                    'step_type': type(step).__name__,
                    'details': step.get_details(),
                    'paramset_values': paramset.values_dict.copy(),
                    'error': str(e),
                    'is_condition': False
                }
                self.errors.append(error_info)
                print(f"Error in execution of {error_info['step_type']} at step {step_idx}: {str(e)}")
                if self.error_handling == 'exit':
                    raise RuntimeError(f"Error executing step: {str(e)}") from e
                elif self.error_handling == 'skip_leaf':
                    self._end_sequence(paramset)
                    return
                elif self.error_handling == 'continue_leaf':
                    new_paramsets = (copy.deepcopy(paramset),)

            step_runtime = time.time() - t0
        
            # Handle branch termination
            if new_paramsets is None:
                self._end_sequence(paramset)
                return

            # Validate new paramsets
            if not isinstance(new_paramsets, tuple) or not all(isinstance(ps, ParameterSet) for ps in new_paramsets):
                raise ValueError(f"Invalid step result, Expected tuple of ParameterSet, got {type(new_paramsets)}")
        
        except Exception as e:
            step_type = type(step).__name__
            details = step.get_details()
            raise RuntimeError(f"Error in {step_type} step at index {step_idx} with details {details}: {str(e)}") from e

        # Process next step for each new parameter set (recursively)
        for ps in new_paramsets:
            ps.update_param('runtime', value=ps.get_value('runtime', 0.0) + step_runtime) # Update cumulative runtime
            self._process(ps, step_idx + 1)

    def _end_sequence(self, paramset: ParameterSet) -> None:
        """Record data for a completed analysis leaf (the end of a branch)."""
        self.finished_leaves += 1
        leaf_dict = paramset.values_dict.copy()
        leaf_dict['datetime'] = datetime.now()
        self.leaf_data.append(leaf_dict)

        # Update progress bar
        elapsed_time = time.time() - self.t0
        if self.finished_leaves > 0 and self.total_leaves > 0:
            eta = (elapsed_time / self.finished_leaves) * (self.total_leaves - self.finished_leaves)
        else:
            eta = 0.0
        progress_desc = (
            f"{self.finished_leaves}/{self.total_leaves} leaves ({self.finished_leaves/self.total_leaves*100:.1f}%) "
            f"| Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s"
        )
        if self.show_notebook_progress_bar:
            self.progress_bar.set_description(progress_desc)
            self.progress_bar.update(1)

