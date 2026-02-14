from functioneer import AnalysisModule

# Rosenbrock function (known minimum: value=0 @ x=1, y=1, a=1, b=100)
def rosenbrock_2d(x, y, a, b):
    return (a - x)**2 + b * (y - x**2)**2

def test_analysis_module():
    anal = AnalysisModule()

    anal.add.define({'a': 1, 'b': 100}) # define a and b
    anal.add.fork('x', (0, 1, 2)) # Fork analysis, create branches for x=0, x=1, x=2
    anal.add.fork('y', (1, 10))
    anal.add.evaluate(func=rosenbrock_2d) #
    results = anal.run()
    print('Example 1 Output:')
    print(results['df'][['a', 'b', 'x', 'y', 'rosenbrock_2d']])

    assert len(results['df']) == 6
    assert results['df']['x'].tolist() == [0, 0, 1, 1, 2, 2]
    assert results['df']['y'].tolist() == [1, 10, 1, 10, 1, 10]
    assert results['df']['rosenbrock_2d'].tolist() == [101, 10001, 0, 8100, 901, 3601]


