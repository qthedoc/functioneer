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

        # Validate name
        if not isinstance(name, str):
            raise TypeError(f"Invalid AnalysisModule: 'name' must be a string, got {type(name)}")
        self.name = name

        # Validate init_param_values
        if not isinstance(init_param_values, dict):
            raise ValueError(f"Invalid AnalysisModule: 'init_param_values' must be a dictionary, got {type(init_param_values)}")
        
        # Validate parameter IDs
        try:
            for param_id in init_param_values:
                Parameter.validate_id(param_id)
        except ValueError as e:
            raise ValueError(f"Invalid Initial Parameters: {str(e)}") from e

        # Analysis Setup
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

        pass

    class AddNamespace:
        """Namespace for adding different types of analysis steps."""

        def __init__(self, parent):
            self.parent: AnalysisModule = parent
            self.define = self.DefineNamespace(parent)
            self.fork = self.ForkNamespace(parent)
            self.optimize = self.OptimizeNamespace(parent)
            self.execute = self.ExecuteNamespace(parent)
            # self.end = self.EndNamespace(parent)

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
                param_id: str,
                value = None,
                condition = None
            ):
                self.parent.sequence.append(Define(param_id, value, condition))

        class ForkNamespace:
            def __init__(self, parent):
                self.parent: AnalysisModule = parent

            def __call__(self, 
                param_id: str,
                value_set: tuple, # value_sets
                condition: Optional[Callable[[], bool]] = None
            ):
                self.parent.sequence.append(Fork(param_id, value_set, condition))

            def multi(self,
                param_ids: tuple[str, ...],
                value_sets: tuple[tuple, ...],
                condition: Optional[Callable[[], bool]] = None
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
                assign_to: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
                unpack_result: bool = False,
                condition: Optional[Callable[[], bool]] = None
            ):
                # self.parent.sequence.append(Execute(func, output_param_ids, input_param_ids, condition))
                self.parent.sequence.append(Execute(
                    func,
                    assign_to,
                    unpack_result,
                    condition                    
                    ))

        class OptimizeNamespace:
            def __init__(self, parent):
                self.parent: AnalysisModule = parent

            def __call__(self, 
                func: Callable,
                assign_to: Optional[str] = None,
                opt_param_ids: Optional[Tuple[str, ...]] = None,
                direction: str = 'min',
                optimizer: Union[str, Callable] = 'SLSQP',
                tol: Optional[float] = None,
                bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                options: Optional[Dict[str, Any]] = None,
                condition: Optional[Callable] = None,
                **kwargs
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
                    tol,
                    bounds,
                    options,
                    condition,
                    **kwargs
                ))

        # class EndNamespace:
        #     def __init__(self, parent):
        #         self.parent: AnalysisModule = parent

        #     def __call__(self, *args, **kwds):
        #         pass

    def run(self,
            # create_pandas = True,
            # verbose = True
        )  -> Dict[str, Any]:
        """Execute the analysis sequence and return results including a DataFrame of leaf data."""
        # self.sequence = tuple(self.sequence)

        self.t0 = time.time()
        self.finished_leaves = 0
        self.leaf_data = []  # Reset leaf data
        try:
            self._process(self.init_paramset, step_idx=0) # start on step 0
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {str(e)}") from e
        finally:
            self.runtime = time.time() - self.t0

        # print('done with analysis!')

        # Create DataFrame from collected leaf data
        df = pd.DataFrame(self.leaf_data) if self.leaf_data else pd.DataFrame()

        results = dict(
            df = df,
            runtime = self.runtime,
            finished_leaves = self.finished_leaves,
        )

        return results

    def _process(self, paramset: ParameterSet, step_idx:int):
        """
        Run a single step of the analysis sequence and recursively process the next step.

        This method evaluates the condition of the current step (if any), executes the step if the
        condition is met, and recursively processes the resulting parameter sets for the next step.
        If the step returns None, the branch terminates, and the current parameter set is recorded.
        Runtime is tracked and accumulated in each parameter set.

        Parameters
        ----------
        paramset : ParameterSet
            The current parameter set for the analysis branch.
        step_idx : int
            The index of the current step in the sequence.

        Returns
        -------
        None
            This method does not return a value; it updates the internal results by calling _end_sequence
            or recursively processing subsequent steps.

        Raises
        ------
        ValueError
            If the step index is invalid, a step returns an invalid output, or a condition/step fails
            due to invalid inputs (e.g., missing parameters, type mismatches).
        AnalysisError
            If an error occurs during condition evaluation or step execution, including step type,
            index, details, and the original exception.
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
        """Record data for a completed analysis leaf."""
        self.finished_leaves += 1

        # Store paramset values with _datetime
        leaf_dict = paramset.values_dict.copy()
        leaf_dict['datetime'] = datetime.now()
        self.leaf_data.append(leaf_dict)

