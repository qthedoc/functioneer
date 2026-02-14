# Functioneer

**Announcement:** Functioneer has just entered public testing and I'd love your feedback on anything: how you're using it, the API, the docs, requested features, etc. Please reach out, I'd love to hear from you. -Quinn

**Author**: Quinn Marsh\
**GitHub**: https://github.com/qthedoc/functioneer/ \
**PyPI**: https://pypi.org/project/functioneer/ 

Functioneer is the ultimate batch runner. Prepare to be an analysis ninja, effortlessly exploring your functions. In just a few lines of code, you can set up thousands or even millions of tests and optimizations on your function. Perfect for parameter sweeps, engineering simulations, digital twin optimization and much more.

## Quick Start

**Install:**
```
pip install functioneer
```

**Full set of examples**: [Examples.ipynb (nbviewer.org)](https://nbviewer.org/github/qthedoc/functioneer/blob/main/examples/Examples.ipynb)*\
*This is currently the main form of documentation.

### Example 1: Forks and Function Evaluation (The Basics)

**Goal**: Test `rosenbrock` function with multiple values for parameters `x` and `y`.

Note: forks for `x` and `y` create a 'grid' of values\
Note: Parameter IDs MUST match your function's args, function evals inside functioneer are fully keyword arg based.
```
from functioneer import AnalysisModule

# Rosenbrock function (known minimum of 0 at: x=1, y=1, a=1, b=100)
def rosenbrock_2d(x, y, a, b):
    return (a-x)**2 + b*(y-x**2)**2

analysis = AnalysisModule() # Create new analysis
analysis.add.define({'a': 1, 'b': 100}) # define a and b
analysis.add.fork('x', (0, 1, 2)) # Fork analysis, create branches for x=0, x=1, x=2
analysis.add.fork('y', (1, 10))
analysis.add.evaluate(func=rosenbrock_2d) #
results = analysis.run()
print('Example 1 Output:')
print(results['df'][['a', 'b', 'x', 'y', 'rosenbrock_2d']])
```
```
Example 1 Output:
   a    b  x   y  rosenbrock_2d
0  1  100  0   1            101
1  1  100  0  10          10001
2  1  100  1   1              0
3  1  100  1  10           8100
4  1  100  2   1            901
5  1  100  2  10           3601
```

### Example 2: Optimization

**Goal**: Optimize `x` and `y` to find the minimum `rosenbrock` value for various `a` and `b` values.

Note: values for `x` and `y` before optimization are used as initial guesses
```
from functioneer import AnalysisModule

def rosenbrock_2d(x, y, a, b):
    return (a-x)**2 + b*(y-x**2)**2

analysis = AnalysisModule({'x': 0, 'y': 0})
analysis.add.fork('a', (1, 2))
analysis.add.fork('b', (0, 100, 200))
analysis.add.optimize(func=rosenbrock_2d, opt_param_ids=('x', 'y'))
results = analysis.run()
print('\nExample 2 Output:')
print(results['df'][['a', 'b', 'x', 'y', 'rosenbrock_2d']])
```
```
Example 2 Output:
   a    b         x         y    rosenbrock_2d
0  1    0  1.000000  0.000000     4.930381e-32
1  1  100  0.999763  0.999523     5.772481e-08
2  1  200  0.999939  0.999873     8.146869e-09
3  2    0  2.000000  0.000000     0.000000e+00
4  2  100  1.999731  3.998866     4.067518e-07
5  2  200  1.999554  3.998225     2.136755e-07
```
## Key Features

- **Quickly test variations of a parameter with a single line of code:** Avoid writing deeply nested loops. Typically varying *n* parameters requires *n* nested loops... not anymore!

- **Quickly setup optimization:** Most optimization libraries require your function to take in and spit out a list or array, BUT this makes it very annoying to remap your parameters to and from the array each time you simple want to add/rm/swap an optimization parameter!

- **Get results in a consistent easy to use format:** No more questions, the results are presented in a nice clean pandas data frame every time.

## Use cases

- **Analysis and Optimization of Digital Twins**: Explore the design trade-space and understand performance of your simulated system.
- **Machine Learning and AI**: Autonomously test thousands of architectures or other parameters for ML models (like neural networks) to see which perform best.
- **Your Imagination is the Limit**: What function will you engineer?

## How Functioneer Works

At its core, functioneer organizes analyses as a tree where a *set of parameters* starts at the trunk and moves out towards the *leaves*. Along the way, the *set of parameters* 'flows' through a series of *analysis steps* (each of which can be defined in a single line of code). Each *analysis step* can modify or use the parameters in various ways, such as defining new parameters, modifying/overwriting parameters, or using the parameters to evaluate or even optimize any function of your choice. One key feature of functioneer is the ability to introduce *forks*: a type of *analysis step* that splits the analysis into multiple parallel *branches*, each exploring different values for a given parameter. Using many *Forks* in series allows you to queue up thousands or even millions of parameter combinations with only a few lines of code. This structured approach enables highly flexible and dynamic analyses, suitable for a wide range of applications.

Summary of most useful types of *analysis steps*:
- Define: Adds a new parameter to the analysis
- Fork: Splits the analysis into multiple parallel *branches*, each exploring different values for a specific parameter
- Evaluate: Calls a provided function using the parameters
- Optimize: Quickly set up an optimization by providing a function and defining which parameters are going to be optimized

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

## Inspiration
I wanted to be an Analysis Ninja... effortlessly swapping parameters and optimization variables and most importantly getting results quickly! But manually rearranging code for what seemed like simple asks was really baking my noodle. Simple things like adding a variable to the analysis, or swapping out an optimization variable, required a shocking amount of code rework. Thus Functioneer was born.

## Acknowledgments
Thanks to the amazing open source communities: Python, numpy, pandas, etc that make this possible.

Thanks to LightManufacturing, where I had the opportunity to develop advanced digital twins for solar thermal facilities... and then analyze them. It was here, where the seed for Functioneer was planted. 

Thank you God for beaming down what seemed like the craziest idea at the time: to structure an analysis as a pipeline of *analysis steps* with the *parameters* flowing thru like water.

## Dev
If anyone wants to help develop Functioneer, there are issues on GitHub with planned features and a dev_notes folder containing possibly useful chicken scratch.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

You are free to use, modify, and distribute this software. Please include proper attribution by retaining the copyright notice in your copies or substantial portions of the software.

