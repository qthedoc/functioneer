# Functioneer

**Author**: Quinn Marsh  
**Date**: January 08, 2025

Functioneer is a Python package that automates the analysis of ANY function, enabling you to test and optimize with unlimited combinations of parameters. Whether you're performing parameter sweeps, sensitivity testing, or optimizing digital twins, Functioneer lets you queue up thousands or even millions of tests in seconds. Easily retrieve and analyze results in formats like pandas for seamless integration into your analysis workflows.

## Use cases

- **Analysis and Optimization of Digital Twins**: Explore the design trade-space and understand performance of your simulated system.
- **Machine Learning and AI**: Autonomously test thousands of architectures or other parameters for ML models (like neural networks) to see which perform best.
- **Your Imagination is the Limit**: What function will you engineer?

## How Functioneer Works

Functioneer is a powerful system for defining and executing complex analysis pipelines. At its core, the toolkit organizes analyses as pipelines, where a set of *parameters* flows sequentially through a series of *analysis steps*. These steps modify the parameters in various ways, such as defining new parameters, updating existing parameter values, or performing operations like function evaluation and optimization. One of the key features of functioneer is the ability to introduce *forks*, which split the analysis into multiple *branches*, each exploring different values for a specific parameter. This structured approach enables highly flexible and dynamic analyses, suitable for a wide range of applications.

### Terms
#### AnalysisModule
* Definition: The central container for an analysis pipeline.
* Function: Holds a sequence of analysis steps and manages a set of parameters that flow through the pipeline.

#### Parameters
* Definition: Named entities that represent inputs, intermediate values, or outputs of the analysis.
* Function: Can be created, modified, or used in computations during analysis steps.

#### Analysis Steps
* Definition: Individual operations performed during the analysis.
* Function: Modify parameters by defining new ones, updating existing values, forking the analysis, or executing/optimizing functions.

#### Fork
* Definition: A special type of analysis step that splits the pipeline into multiple branches.
* Function: Creates independent branches where each branch explores a different value or configuration for a given parameter.

#### Branch
* Definition: One of the independent paths created by a Fork.
* Function: Represents a distinct variation of the analysis, each processing a specific set of parameter values.

#### Leaf
* Definition: The endpoint of a branch after all analysis steps have been executed.
* Function: Represents the final state of parameters for that branch. Each leaf corresponds to a specific combination of parameter values and results. When results are tabulated, each row corresponds to a leaf.

## Installation

Install Functioneer directly from PyPI:

```
pip install functioneer
```

## Getting Started
Here's a few quick examples of how to use `functioneer`. Each example will build on the last, introducing one piece of functionality. By the end you will have witnessed the computational power of this fully armed and fully operational library.

### Choosing a Function to Analyze
Functioneer is designed to analyze ANY function(s) with ANY number of inputs and outputs. For the following examples, the [Rosenbrock Function](https://en.wikipedia.org/wiki/Rosenbrock_function) is used for its relative simplicity, 4 inputs (plenty to play with) and its historical significance as an optimization benchmark.

```
# Rosenbrock function (known minimum of 0 at: x=1, y=1, a=1, b=100)
def rosenbrock(x, y, a, b):
    return (a-x)**2 + b*(y-x**2)**2
```

### Example 1: The Basics (Defining Parameters and Executing a Function)
Set up an *analysis sequence* by defining four parameters (the inputs needed for the rosenbrock function), then executing the function (with parameter ids matched to kwargs)

```
import functioneer as fn

# Create new analysis
anal = fn.AnalysisModule() # its not ānal is anál!

# Define analysis sequence
anal.add(fn.Define('a', 1)) # Define parameter 'a'
anal.add(fn.Define('b', 100)) # Define parameter 'b'
anal.add(fn.Define('x', 1)) # Define parameter 'x'
anal.add(fn.Define('y', 1)) # Define parameter 'y'
anal.add(fn.Execute(func=rosenbrock, output_param_ids='rosen')) # Execute function with parameter ids matched to kwargs

# Run the analysis sequence
results = anal.run()

print(results['df'])
```

Note: the `results['df']` is a pandas DataFrame with runtime and datetime fields included
```
Output:
   runtime  a    b  x  y  rosen                   datetime
0      0.0  1  100  1  1      0 2025-01-03 17:06:21.252981
```

As predicted, the `rosen` parameter evaluates to 0 when a=1, b=100, x=1, y=1

But lets say you want to test a range of values for some parameters...

### Example 2: Single Parameter Forks (Testing Variations of a Parameter)
If you want to test a set of values for a parameter you can create a *fork* in the *analysis sequence*, resulting in several *analysis branches*. In each *analysis branch*, the respective parameter will be set to one the values in the set.

Say we want to evaluate and plot the rosenbrock surface over the x-y domain. Lets evaluate a grid where x=(0, 1, 2) and y=(1, 10) which should result in 6 total points...
```
# Create new analysis
anal = fn.AnalysisModule()

# Define analysis sequence
anal.add(fn.Define('a', 1)) # Define parameter 'a'
anal.add(fn.Define('b', 100)) # Define parameter 'b'
anal.add(fn.Fork('x', value_sets=(0, 1, 2))) # Fork analysis, create a branch for each value of 'x': 0, 1, 2
anal.add(fn.Fork('y', value_sets=(1, 10))) # Fork analysis, create a branch for each value of 'y': 1, 10
anal.add(fn.Execute(func=rosenbrock, output_param_ids='rosen')) # Execute function (for each branch) with parameters matched to kwargs

# Run the analysis sequence
results = anal.run()
print(results['df'].drop(columns='datetime'))
```
```
Output:
    runtime  a    b  x   y  rosen
0  0.000994  1  100  0   1    101
1  0.000994  1  100  0  10  10001
2  0.000994  1  100  1   1      0
3  0.000994  1  100  1  10   8100
4  0.000994  1  100  2   1    901
5  0.000994  1  100  2  10   3601
```
The parameters `x` and `y` were given 3 and 2 fork values respectively, this created 6 total *leaves* (end of each branch) in the analysis. `rosen` has been evaluated for each *leaf*. Essentially you have begun to map the Rosenbrock function over the x-y domain.

### Example 3: Optimization
Lets say you want to find the local minimum of the Rosenbrock (optimize `x` and `y`) for several different flavors rosenbrock functions (each with different `a` nnd `b` parameters)...
```
# Create new analysis
anal = fn.AnalysisModule()

# Define analysis sequence
anal.add(fn.Fork('a', value_sets=(1, 2))) # Fork analysis, create a branch for each value of 'a': 0, 1, 2
anal.add(fn.Fork('b', value_sets=(0, 100, 200))) # Fork analysis, create a branch for each value of 'b': 0, 100, 200
anal.add(fn.Define('x', 0))
anal.add(fn.Define('y', 0))

anal.add(fn.Optimize(func=rosenbrock, obj_param_id='rosen', opt_param_ids=('x', 'y')))

# Run the analysis sequence
results = anal.run()
print(results['df'].drop(columns='datetime'))
```
```
Output:
    runtime  a    b         x         y         rosen 
0  0.001017  1    0  1.000000  0.000000  4.930381e-32 
1  0.009276  1  100  0.999763  0.999523  5.772481e-08 
2  0.007347  1  200  0.999939  0.999873  8.146869e-09 
3  0.002572  2    0  2.000000  0.000000  0.000000e+00 
4  0.011093  2  100  1.999731  3.998866  4.067518e-07 
5  0.030206  2  200  1.999554  3.998225  2.136755e-07 
```
For each branch, the Rosenbrock Function has been minimized and the solution values for 'x', 'y' and 'rosen' are shown.

Note: the initial values used in the optimization are just the existing parameter values (in this case x and y are 0).

Note: due to optimization the runtimes for some of the analyses have gone up.

### Example 4: Multi-parameter Forks
If you want to test specific combinations of parameters (instead of creating a grid) use a *multi-parameter fork*. The following will result in 3 branches: (a=0, b=0), (a=1, b=100), (a=2, b=200)
```
fn.Fork(('a', 'b'), value_sets=((0, 1, 2), (0, 100, 200)))
```
### Example 5: Conditional Analysis Step's
Any analysis step can be given a conditional function that must return true at runtime or else the analysis step will be skipped. One use case for this is when you want to skip an expensive analysis step if the parameters aren't looking good.

As an arbitrary example, assume that we only care about cases where the optimized value of `y` is above 0.5. Also assume `expensive_func` is costly to run and we want to avoid running it when `y<0.5`. 
```
# Create new analysis
anal = fn.AnalysisModule()

# Define analysis sequence
anal.add(fn.Fork('a', value_sets=(0, 1, 2)))
anal.add(fn.Fork('b', value_sets=(0, 100, 200)))
anal.add(fn.Define('x', 0))
anal.add(fn.Define('y', 0))
anal.add(fn.Optimize(func=rosenbrock, obj_param_id='rosen', opt_param_ids=('x', 'y')))

# Only evaluate 'expensive_func' if the optimized 'y' is above 0.5
expensive_func = lambda x, y: x+y
anal.add(fn.Execute(func=expensive_func, output_param_ids='expensive_out', condition=lambda y: y>0.5))

# Run the analysis sequence
results = anal.run()
print(results['df'].drop(columns='datetime'))
```
```
Output:
    runtime  a    b         x         y         rosen  expensive_param  
0  0.004001  1    0  1.000000  0.000000  4.930381e-32              NaN  
1  0.009702  1  100  0.999763  0.999523  5.772481e-08         1.999286  
2  0.017009  1  200  0.999939  0.999873  8.146869e-09         1.999811  
3  0.000995  2    0  2.000000  0.000000  0.000000e+00              NaN  
4  0.016001  2  100  1.999731  3.998866  4.067518e-07         5.998596  
5  0.020995  2  200  1.999554  3.998225  2.136755e-07         5.997779  
```
Notice how the evaluation of `expensive_param` has been skipped where `y` did not meet the criteria `y>0.5`

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

You are free to use, modify, and distribute this software. Please include proper attribution by retaining the copyright notice in your copies or substantial portions of the software.

