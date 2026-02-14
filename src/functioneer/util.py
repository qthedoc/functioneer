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

import inspect
from typing import Any, Callable, Dict

def call_with_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Any:
    '''
    Calls a function with matching kwargs, passing all provided kwargs if the function has **kwargs,
    while checking for missing required arguments. Functions with *args are not supported and raise an error.

    Args:
        func: The function to call.
        kwargs: Dictionary of keyword arguments to pass.

    Returns:
        The result of the function call.

    Raises:
        ValueError: If required arguments (without defaults) are missing or if the function has *args.
    '''
    # Get the function signature
    signature = inspect.signature(func)
    parameters = signature.parameters

    # Check for *args (VAR_POSITIONAL)
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters.values()):
        raise ValueError(
            f"Function '{func.__name__}' has a *args parameter, which is not supported by call_with_kwargs. "
            "Use keyword arguments or **kwargs instead."
        )

    # Check for missing required arguments
    missing_args = []
    for name, param in parameters.items():
        if (
            param.default == inspect.Parameter.empty
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and name not in kwargs
        ):
            missing_args.append(name)

    # Raise an error if required arguments are missing
    if missing_args:
        missing_args_str = [f"'{arg}'" for arg in missing_args]
        raise ValueError(
            f"Missing required parameters for function '{func.__name__}': {', '.join(missing_args_str)}"
        )

    # Check if the function has a **kwargs parameter
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    if has_kwargs:
        # Pass all kwargs, including those matching named parameters and extras
        return func(**kwargs)
    else:
        # Pass only kwargs that match the function's parameters
        matched_kwargs = {name: kwargs[name] for name in kwargs if name in parameters}
        return func(**matched_kwargs)