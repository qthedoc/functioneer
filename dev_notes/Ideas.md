


## Integration of computation (functioneer) and presentation (plotly)
Analysis has two parts: the computation and the presentation.
Inherently people only care about the presentation! the plot, the chart, the result! 
Thats why we do analysis! The computation (and any other preliminary step) requires thinking backwards, which slows us down!
Currently functioneer only helps with the computation (getting a big ass table of data).
But it should be possible to combine the setting up of the computation AND the presentation.
For example: When I define a fork I probably already know that fork is going to be my X axis on a line plot.

More examples:
- set a fork as plot x-axis
- set a fork to be multiple series in a plot
- set up a mesh grid for a surface plot
- set a fork to be aggregated (mean, std, mean /w std error bars, max, min)
- set an evaluation to be set as error bars

Functioneer Output:
- By default I feel functioning should remain as just a big ass table generator but we should make these plotting options very available.
- I see that pandas contains a plotly back end, maybe functioneer should also have integrated plotly functionality.
- To enable more programmatic usage of functioneer (not for visual analysis), we should include aggregated data in simpler data types (list,  dict) in the results. Pandas is a bit heavy for programmatic usage. Maybe have a option to turn off pandas.

