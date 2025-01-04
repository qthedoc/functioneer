# Functioneer

**Author**: Quinn Marsh  
**Date**: January 03, 2025 

Functioneer is a powerful Python package that simplifies the setup of automated analyses for your functions. With Functioneer, you can configure and execute functions with unlimited combinations of inputs, making it perfect for optimizations, parameter sweeps, testing, and any scenario where you need to test a function(s) over a wide range of inputs.

Great for exploring trade space and testing digital twins

## Features

- **Fast Setup**: Define inputs, configurations, and optimizations in seconds.
- **Automated Execution**: Run functions with diverse parameter combinations, reducing repetitive setup.
- **Flexible Input Handling**: Supports a variety of input types for complex analyses.
- **Combinatorial Efficiency**: Specify ranges and conditions to create the full space of function possibilities.
- **Output Management**: Easily retrieve and analyze the output of all function executions.

## Installation

Install Functioneer directly from PyPI:

```
pip install functioneer
```

## Examples
Functioneer is designed to analyse ANY function; for the following examples, the [Rosenbrock Function](https://en.wikipedia.org/wiki/Rosenbrock_function) is used for its relative simplicity, 4 inputs (plenty to play with) and its historical significance as an optimization benchmark.
```
# Rosenbrock function (known minimum of 0 at: x=1, y=1, a=1, b=100)
def rosenbrock(x, y, a, b):
    return (a-x)**2 + b*(y-x**2)**2
```
### Defining Parameters and Executing a Function
```
import functioneer as fn

# Create new analysis
anal = fn.AnalysisModule()

# Define analysis sequence
anal.add(fn.Define('a', 1)) # Deine parameter 'a'
anal.add(fn.Define('b', 100)) # Deine parameter 'b'
anal.add(fn.Define('x', 1)) # Deine parameter 'x'
anal.add(fn.Define('y', 1)) # Deine parameter 'y'
anal.add(fn.Execute(func=rosenbrock, output_param_ids='rosen')) # Execute function with parameters matched to kwargs

# Run thru the analysis sequence
results = anal.run()

print(results['df'])
```
```
Ouputs:
   runtime  a    b  x  y  rosen                   datetime
0      0.0  1  100  1  1      0 2025-01-03 17:06:21.252981
```
As predicted, 'rosen' parameter evaluates to 0 when x=1, y=1, a=1, b=100

But lets say you want to test a range of values on some parameters...

### Forking Parameter: testing multiple variations of a parameter
```
# Create new analysis
anal = fn.AnalysisModule()

# Define analysis sequence
anal.add(fn.Define('a', 1)) # Deine parameter 'a'
anal.add(fn.Define('b', 100)) # Deine parameter 'b'
anal.add(fn.Fork('x', value_sets=(0, 1, 2))) # Fork analysis, create a branch for each value of 'x': 0, 1, 2
anal.add(fn.Fork('y', value_sets=(1, 10))) # Fork analysis, create a branch for each value of 'y': 1, 10
anal.add(fn.Execute(func=rosenbrock, output_param_ids='rosen')) # Execute function with parameters matched to kwargs

# Run thru the analysis sequence
results = anal.run()
print(results['df'])
```
```
Outputs:
    runtime  a    b  x   y  rosen                   datetime
0  0.000994  1  100  0   1    101 2025-01-03 22:06:50.916681
1  0.000994  1  100  0  10  10001 2025-01-03 22:06:50.918690
2  0.000994  1  100  1   1      0 2025-01-03 22:06:50.919681
3  0.000994  1  100  1  10   8100 2025-01-03 22:06:50.919681
4  0.000994  1  100  2   1    901 2025-01-03 22:06:50.920680
5  0.000994  1  100  2  10   3601 2025-01-03 22:06:50.920680
```
The parameters 'x' and 'y' were given 3 and 2 fork values respectively, this created 6 final heads in the analysis. 'rosen' has been evaluated for each combination. Essentially you have begun to map the Rosenbrock function for the case a=1 b=100.

But lets say you want to find the local minima of the Rosenbrock (optimize 'x' and 'y') for several different flavors rosenbrocks (each with different 'a' nnd 'b' parameters)...

### Optimization

### Conditionals


## License
Functioneer is released as open-source software with a custom license. Please see the LICENSE file for usage guidelines.