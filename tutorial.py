import functioneer as fn

# Example functions

# Rosenbrock function (known minimum at: x=1, y=1 for a=1, b=100)
def rosenbrock(x, y, a, b):
    return (a-x)**2 + b*(y-x**2)**2



# Basic Functioneer Demo

# test rosenbrock function (known minimum at: x=1, y=1 for a=1, b=100)
def rosenbrock(x, y, a, b):
    return (a-x)**2 + b*(y-x**2)**2

# Create new analysis
anal = fn.AnalysisModule()

# Define analysis sequence
anal.add(fn.Define('a', 1)) # Deine parameter 'a'
anal.add(fn.Define('b', 100)) # Deine parameter 'b'
anal.add(fn.Define('x', 1)) # Deine parameter 'x'
anal.add(fn.Define('y', 1)) # Deine parameter 'y'
anal.add(fn.Execute(func=rosenbrock, output_param_ids='rosen')) # Execute function with parameters matched to kwargs

# Run analysis
results = anal.run()

print(results['df'])