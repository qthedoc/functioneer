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
from typing import Any, Callable

from functioneer.util import call_with_kwargs

class Parameter():
    def __init__(self,
            id: str,
            value = None,
            # value_set = None, 
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
            
    RESERVED_IDS = {'runtime', 'datetime'}

    @staticmethod
    def validate_id(id: Any) -> None:
        """
        Validate a parameter ID.

        A valid ID is a non-empty string that is not a reserved name (e.g., '_runtime', '_datetime').

        Parameters
        ----------
        id : Any
            The parameter ID to validate.

        Raises
        ------
        ValueError
            If the ID is not a non-empty string or is a reserved name.

        Examples
        --------
        >>> Parameter.validate_id('x')
        >>> Parameter.validate_id('_runtime')
        ValueError: Invalid parameter ID: '_runtime' is a reserved name. Choose a different name.
        """
        if not isinstance(id, str) or id == '':
            raise ValueError(f"Invalid parameter ID: '{id}' must be a non-empty string")
        if id in Parameter.RESERVED_IDS:
            raise ValueError(f"Invalid parameter ID: '{id}' is a reserved name. Choose a different name.")
    
    @staticmethod
    def validate_id_iterable(ids: Any) -> None:
        """
        Validate an iterable of parameter IDs.

        Parameters
        ----------
        ids : Any
            An iterable of parameter IDs to validate.

        Raises
        ------
        ValueError
            If ids is not an iterable of non-empty strings or contains reserved names.
        """
        if not hasattr(ids, '__iter__') or isinstance(ids, str):
            raise ValueError(f"Invalid parameter IDs: '{ids}' must be an iterable of strings, not a string")
        for id in ids:
            Parameter.validate_id(id)
            
class ParameterSet(dict[str, Parameter]):
    def __init__(self):
        self.add_param(Parameter('runtime', 0))

    def add_param(self, parameter):
        if not isinstance(parameter, Parameter):
            raise ValueError("parameter must be an instance of Parameter")
        
        id = parameter.id

        if id in self:
            logging.info(f"Parameter '{id}' overwritten")

        self[id] = parameter

    def get_value(self, param_id: str, default=None):
        parameter = self.get(param_id)
        value = parameter.value if (parameter and parameter.value is not None) else default
        return value 

    @property
    def values_dict(self):
        return {key: param.value for key, param in self.items()}
    
    def update_param(self, id, value=None):
        """
        Updates attributes of parameter matching 'id'
        Creates a new parameter if no match

        TODO add other attributes here if needed
        """ 
        if value is not None:
            if id in self:
                self[id].value = value
            else:
                self.add_param(Parameter(id, value))

    def update_param_values(self, values_dict: dict):
        """
        Updates values of existing parameters. 
        Adds new parameter if key does not match existing.
        """
        for id, val in values_dict.items():
            self.update_param(id, val)

    # def get_value_list(self, param_ids):
    #     # TODO Validate param_ids are in paramset
    #     return [self[id].value for id in param_ids]

    def call_with_kwargs(self, func: Callable) -> Any:
        """
        Calls a function with parameter values from the ParameterSet, using matching keyword args.
        Supports functions with **kwargs and checks for missing required arguments.
        Functions with *args are not supported and raise an error.

        Args:
            func: The function to call.

        Returns:
            The result of the function call.

        Raises:
            ValueError: If required arguments are missing or if the function has *args.
        """
        return call_with_kwargs(func, self.values_dict)



