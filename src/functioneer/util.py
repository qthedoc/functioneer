import inspect


def call_with_matched_kwargs(func, kwargs):
    '''
    calls funciton with matching kwargs, throws error if required arg is missing.
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
        # TODO: catch this error higher up so we can provide info about WHAT param or funciton was being evaluated
        raise ValueError(f"Missing the following required params while evaluating function: {', '.join(missing_args)}")

    # Call the function with matched kwargs
    return func(**matched_kwargs)