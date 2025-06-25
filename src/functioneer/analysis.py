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
import pandas as pd
from datetime import datetime
import time

from functioneer.steps import AnalysisStep, Define, Fork, Execute, Optimize
from functioneer.parameter import ParameterSet, Parameter
from functioneer.util import call_with_matched_kwargs

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
    def __init__(self, init_param_values={}, name='') -> None:
        """ Initialize Functioneer Analysis

        Args:
            init_param_values : dict, optional
                Initial parameter values as a dictionary with parameter IDs (str) as keys and their values.
                Parameter IDs must be valid (non-empty strings, not reserved names like 'runtime' or 'datetime').
                Defaults to an empty dictionary.
            name : str, optional
                Name of the analysis module. Defaults to an empty string.

        Raises:
            ValueError: If init_param_values is not a dictionary, contains invalid parameter IDs, or includes reserved names.
            TypeError: If name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError(f"Invalid AnalysisModule: 'name' must be a string, got {type(name)}")
        if not isinstance(init_param_values, dict):
            raise ValueError(f"Invalid AnalysisModule: 'init_param_values' must be a dictionary, got {type(init_param_values)}")
        for param_id in init_param_values:
            Parameter.validate_id(param_id)

        self.name = name
        self.sequence: list[AnalysisStep] = []
        self.init_paramset: ParameterSet = ParameterSet()
        self.init_paramset.update_param_values(init_param_values)
        self.init_paramset.add_param(Parameter('runtime', 0))
        self.finished_leaves:int = 0

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
            
        def fork(self, param_or_dict: Union[str, Dict[str, Tuple[Any, ...]]], value_set: Optional[Tuple[Any, ...]] = None, condition: Optional[Callable[..., bool]] = None) -> None:
            """Fork analysis with a single parameter or dictionary of parameter value sets.

            Args:
                param_or_dict: Parameter ID (str) or dictionary of parameter IDs to value sets.
                value_set: Tuple of values for the parameter (if param_or_dict is a string).
                condition: Optional condition function to determine if the step should run.

            Examples:
                >>> anal.add.fork('x', (0, 1, 2))  # Fork single parameter
                >>> anal.add.fork({'x': (0, 1, 2), 'y': (0, 10, 20)})  # Fork multiple parameters

            Raises:
                ValueError: If inputs are invalid (e.g., wrong types, missing value_set, or value_set provided with dict).
            """
            if isinstance(param_or_dict, dict):
                if value_set is not None:
                    raise ValueError(
                        "When forking multiple parameters with a dictionary, the 'value_set' argument is ignored. "
                        "Use either fork(param_id: str, value_set: Tuple[Any, ...]) or fork(params: Dict[str, Tuple[Any, ...]])."
                    )
                self.parent.sequence.append(Fork(param_or_dict, condition))
            elif isinstance(param_or_dict, str) and value_set is not None:
                self.parent.sequence.append(Fork({param_or_dict: value_set}, condition))
            else:
                raise ValueError("Expected (param_id, value_set) or {param_id: value_set, ...}")
            
        def execute(self, func: Callable[..., Any], assign_to: Optional[Union[str, List[str], Tuple[str, ...]]] = None, unpack_result: bool = False, condition: Optional[Callable[..., bool]] = None) -> None:
            """Execute a function and store its result in the ParameterSet.

            Args:
                func: Function to execute, taking parameter values as input.
                assign_to: Custom Parameter ID(s) to store the result (str or list/tuple of strings).
                unpack_result: If True, unpacks a dictionary result into multiple parameters.
                condition: Optional condition function to determine if the step should run.

            Examples:
                >>> anal.add.execute(my_function)  # Execute function, store result
                >>> anal.add.execute(my_function, assign_to='new_param')  # Execute function, store result to param: 'new_param'
                >>> anal.add.execute(my_function_returns_dict, unpack_result=True)  # Unpack dict result
                >>> anal.add.execute(my_function_returns_dict, assign_to=['out_1', 'out_2'], unpack_result=True)  # Unpack only dict keys: out_1, out_2

            Raises:
                ValueError: If func is not callable or assign_to is invalid.
            """
            self.parent.sequence.append(Execute(func, assign_to, unpack_result, condition))
                
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
            # create_pandas = True,
            # verbose = True
        )  -> Dict[str, Any]:
        """Execute the analysis sequence and return results including a DataFrame of leaf data.

        Returns:
            Dict[str, Any]: Dictionary containing the results DataFrame, runtime, and number of finished leaves.
        """
        self.t0 = time.time()
        self.finished_leaves = 0
        self.leaf_data = []  # Reset leaf data
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
            runtime = self.runtime,
            finished_leaves = self.finished_leaves,
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
                run_step = paramset.call_with_matched_kwargs(step.condition) if step.condition else True
            except Exception as e:
                raise RuntimeError(f"Error evaluating step condition: {str(e)}") from e

            # Run the analysis step
            t0 = time.time()
            try:
                new_paramsets = step.run(paramset) if run_step else (copy.deepcopy(paramset),)
            except Exception as e:
                raise RuntimeError(f"Error executing step: {str(e)}") from e
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

