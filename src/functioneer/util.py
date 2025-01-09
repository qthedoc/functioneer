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

def call_with_matched_kwargs(func, kwargs):
    '''
    calls function with matching kwargs, throws error if required arg is missing.
    TODO: could add a special case for a kwarg called kwargs where all kwargs are just fed in there. this allows user to have **kwargs on their functions
    '''
    # Get the function signature
    signature = inspect.signature(func)
    arguments = signature.parameters

    # Match provided kwargs with function arguments
    matched_kwargs = {}
    missing_args = []
    for name, arg in arguments.items():
        if name in kwargs:
            matched_kwargs[name] = kwargs[name]
        elif arg.default == inspect.Parameter.empty:
            missing_args.append(name)

    # Raise a warning if required arguments are missing
    if missing_args:
        missing_args = [f"'{arg}'" for arg in missing_args]
        # TODO: catch this error higher up so we can provide info about WHAT param or function was being evaluated
        raise ValueError(f"Missing the following required params while evaluating function: {', '.join(missing_args)}")

    # Call the function with matched kwargs
    return func(**matched_kwargs)