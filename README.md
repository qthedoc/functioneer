# Functioneer

**Author**: Quinn Marsh  
**Date**: February 02, 2025\
**PyPI**: https://pypi.org/project/functioneer/

Functioneer lets you effortlessly explore function behavior with automated batch analysis. With just a few lines of code, you can queue up thousands or even millions of function evaluations, testing and optimizing with unlimited parameter combinations. Retrieve structured results in formats like pandas for seamless integration into your workflows. Perfect for parameter sweeps, engineering simulations, and digital twin optimization.

## Use cases

- **Analysis and Optimization of Digital Twins**: Explore the design trade-space and understand performance of your simulated system.
- **Machine Learning and AI**: Autonomously test thousands of architectures or other parameters for ML models (like neural networks) to see which perform best.
- **Your Imagination is the Limit**: What function will you engineer?

## How Functioneer Works

At its core, functioneer organizes analyses as pipelines, where a set of *parameters* flows sequentially through a series of *analysis steps*. These *analysis steps* modify the parameters in various ways, such as defining new parameters, modifying parameter values, or performing operations like function evaluation and optimization. One of the key features of functioneer is the ability to introduce *forks*, which split the analysis into multiple *branches*, each exploring different values for a specific parameter. Functioneer *Forks* are what let you queue up thousands or even millions of parameter combinations in only a few lines of code. This structured approach enables highly flexible and dynamic analyses, suitable for a wide range of applications.

<details>
<summary>
<span style="font-size:1.5em;">Important Terms</span>
</summary>

* AnalysisModule
    * Definition: The central container for an analysis pipeline.
    * Function: Holds a sequence of analysis steps and manages a set of parameters that flow through the pipeline.

* Parameters
    * Definition: Named entities that represent inputs, intermediate values, or outputs of the analysis.
    * Function: Can be created, modified, or used in computations during analysis steps.

* Analysis Steps
    * Definition: Individual operations performed during the analysis.
    * Function: Modify parameters by defining new ones, updating existing values, forking the analysis, or executing/optimizing functions.

* Fork
    * Definition: A special type of *analysis step* that splits the pipeline into multiple branches.
    * Function: Creates independent branches where each branch explores a different value or configuration for a given parameter.

* Branch
    * Definition: One of the independent paths created by a Fork.
    * Function: Represents a distinct variation of the analysis, each processing a specific set of parameter values.

* Leaf
    * Definition: The endpoint of a branch after all analysis steps have been executed.
    * Function: Represents the final state of parameters for that branch. Each leaf corresponds to a specific combination of parameter values and results. When results are tabulated, each row corresponds to a leaf.
</details>

## Installation

Install Functioneer directly from PyPI:

```
pip install functioneer
```

## Getting Started
Below are a few quick examples of how to use Functioneer. Each example will build on the last, introducing one piece of functionality. By the end you will have witnessed the computational power of this fully armed and fully operational library.

### Choose a Function to Analyze
Functioneer is designed to analyze ANY function(s) with ANY number of inputs and outputs. For the following examples, the [Rosenbrock Function](https://en.wikipedia.org/wiki/Rosenbrock_function) is used for its relative simplicity, 4 inputs (plenty to play with) and its historical significance as an optimization benchmark.

```
# Rosenbrock function (known minimum of 0 at: x=1, y=1, a=1, b=100)
def rosenbrock(x, y, a, b):
    return (a-x)**2 + b*(y-x**2)**2
```

### Example 1: The Basics (Defining Parameters and Executing a Function)
Set up an *analysis sequence* by defining four parameters (the inputs needed for the Rosenbrock function), then executing the function (with parameter ids matched to kwargs)
Note: Parameter IDs MUST match your function's args

```
import functioneer as fn

# Create new analysis
anal = fn.AnalysisModule() # its not ānal is anál!

# Define analysis sequence
anal.add.define('a', 1) # Define parameter 'a'
anal.add.define('b', 100) # Define parameter 'b'
anal.add.define('x', 1) # Define parameter 'x'
anal.add.define('y', 1) # Define parameter 'y'

anal.add.execute(func=rosenbrock, output_param_ids='rosen') # Execute function with parameter ids matched to kwargs

# Run the analysis sequence
results = anal.run()

print(results['df'])
```

```
Output:
   runtime  a    b  x  y  rosen                   datetime
0      0.0  1  100  1  1      0 2025-01-03 17:06:21.252981
```

As predicted, the `rosen` parameter evaluates to 0 when a=1, b=100, x=1, y=1

Note: the `results['df']` is a pandas DataFrame containing all parameters in addition to *runtime* and *datetime* for the given branch

### Example 2: Single Parameter Forks (Testing Variations of a Parameter)
Let's say you want to test a range of values for some parameters...
If you want to test a set of values for a parameter you can create a *fork* in the *analysis sequence*. This splits the analysis into multiple *branches*, each exploring different values for a the given parameter.

Say we want to evaluate and plot the Rosenbrock surface over the x-y domain. Let's evaluate Rosenbrock on a grid where x=(0, 1, 2) and y=(1, 10) which should result in 6 final *branches* / *leaves*...

Note: some boiler plate can be removed by defining initial parameters in the AnalysisModule() declaration
```
# Create new analysis
init_params = dict(a=1, b=100, x=1, y=1) # initial parameters will be overwritten by forks, optimizations, etc
anal = fn.AnalysisModule(init_params)

# Define analysis sequence
anal.add.fork('x', value_sets=(0, 1, 2)) # Fork analysis, create a branch for each value of 'x': 0, 1, 2
anal.add.fork('y', value_sets=(1, 10)) # Fork analysis, create a branch for each value of 'y': 1, 10

anal.add.execute(func=rosenbrock, output_param_ids='rosen') # Execute function (for each branch) with parameters matched to kwargs

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
Let's say you want to find the local minimum of the Rosenbrock (optimize `x` and `y`) for several variations of `a` and `b` (different flavors Rosenbrock functions). You would fork the analysis at parameters `a` and `b`, then perform an optimization on each branch.
```
# Create new analysis
anal = fn.AnalysisModule(dict(x=0, y=0))

# Define analysis sequence
anal.add.fork('a', value_set=(1, 2)) # Fork analysis, create a branch for each value of 'a': 0, 1, 2
anal.add.fork('b', value_set=(0, 100, 200)) # Fork analysis, create a branch for each value of 'b': 0, 100, 200

anal.add.optimize(func=rosenbrock, obj_param_id='rosen', opt_param_ids=('x', 'y'))

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
For each branch, the Rosenbrock Function has been minimized and the solution values for `x`, `y` and `rosen` are shown.

Note: the initial values (`x0`) used in the optimization are just the existing parameter values (in this case x and y are 0).

Note: due to optimization the runtimes for some of the analyses have gone up.

### Example 4: Multi-parameter Forks
If you want to test specific combinations of parameters (instead of creating a grid) use a *multi-parameter fork*.
```
# Create new analysis
anal = fn.AnalysisModule(dict(a=1, b=100))

# Define analysis sequence
anal.add.fork.multi(('x', 'y'), value_sets=((0, 1, 2), (0, 10, 20))) # Fork analysis, create a branch for each value of 'y': 1, 10

anal.add.execute(func=rosenbrock, output_param_ids='rosen') # Execute function (for each branch) with parameters matched to kwargs

# Run the analysis sequence
results = anal.run()
print(results['df'].drop(columns='datetime'))
```
```
Output:
   runtime  a    b  x   y  rosen
0      0.0  1  100  0   0      1
1      0.0  1  100  1  10   8100
2      0.0  1  100  2  20  25601
```
Notice 3 branches have been create for each combination of `x` and `y`: `(x=0, y=0), (x=1, y=10), (x=2, y=20)`

### Example 5: Analysis Steps can be Conditional
Any *analysis step* can be given a conditional function that must return true at runtime or else the *analysis step* will be skipped. An example use case is when you want to skip an expensive *analysis step* if the parameters aren't looking "good".

As an arbitrary example, assume that we only care about cases where the optimized value of `y` is above 0.5. Also assume `expensive_func` is costly to run and we want to avoid running it when `y<0.5`. 
```
# Create new analysis
anal = fn.AnalysisModule(dict(x=0, y=0))

# Define analysis sequence
anal.add.fork('a', value_set=(1, 2))
anal.add.fork('b', value_set=(0, 100, 200))
anal.add.optimize(func=rosenbrock, obj_param_id='rosen', opt_param_ids=('x', 'y'))

# Only evaluate 'expensive_func' if the optimized 'y' is above 0.5
expensive_func = lambda x, y: x+y
anal.add.execute(func=expensive_func, output_param_ids='expensive_param', condition=lambda y: y>0.5)

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
Notice how the evaluation of `expensive_param` has been skipped where the optimized `y` did not meet the criteria `y>0.5`

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

You are free to use, modify, and distribute this software. Please include proper attribution by retaining the copyright notice in your copies or substantial portions of the software.

