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


import logging
import inspect

class Parameter():
    def __init__(self,
            id: str,
            value = None,
            value_set = None, 
            # fun = None,
            # bounds = (None, None),
        ):
        
        if isinstance(id, str) and id != '':
            self.id = id
        else:
            logging.error(f"Parameter name must be a string with at least one character")
        self.value = value
        # self.values = values
        # self.function = fun
        # self.bounds = bounds

    def set(self, **kwargs):
        '''
        set any attribute of a parameter
        '''
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
            else:
                raise AttributeError(f"Attribute '{attr}' does not exist in the parameter.")
            
    @staticmethod
    def is_valid_id(id):
        # TODO add this to other places
        return isinstance(id, str) and id != ''
    
    @staticmethod
    def is_valid_id_iterable(id_iterable):
        return isinstance(id_iterable, (list, tuple)) and all([Parameter.is_valid_id(id) for id in id_iterable])

            
class ParameterSet(dict):
    def __init__(self):

        self.add_param(Parameter('runtime', 0))

    def add_param(self, parameter):
        if not isinstance(parameter, Parameter):
            raise ValueError("parameter must be an instance of Parameter")
        
        id = parameter.id

        if id in self:
            logging.info(f"Parameter '{id}' overwritten")

        self[id] = parameter

    @property
    def values_dict(self):
        return {key: param.value for key, param in self.items()}
    
    def update_param(self, id, value=None):
        """
        Updates attributes of parameter matching 'id' or creates a new parameter if none match

        TODO add other attributes here if needed
        """ 
        if value is not None:
            if id in self:
                self[id].value = value
            else:
                self.add_param(Parameter(id, value))

    def update_param_values(self, values_dict: dict):
        for id, val in values_dict.items():
            self.update_param(id, val)

    # def get_value_list(self, param_ids):
    #     # TODO Validate param_ids are in paramset
    #     return [self[id].value for id in param_ids]
    
    def call_with_matched_kwargs(self, func):
        """
        Calls function with parameter values from the ParameterSet with matching keyword args.
        Throws error if required arg is missing in paramset.
        """
        kwargs = self.values_dict

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

    def call_with_positional_args(self, func, param_ids):
        """
        Calls function with parameter values as positional args.
        The order of the args is the same as given in 'param_ids'.
        """
        # arg_list = self.get_value_list(self.input_params)

        # TODO Validate param_ids are in paramset
        arg_list = [self[id].value for id in param_ids]
        return func(*arg_list)


