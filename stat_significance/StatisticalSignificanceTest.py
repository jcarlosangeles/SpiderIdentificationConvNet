from scipy.stats import ttest_ind
from pandas import DataFrame
from pandas import read_csv
from matplotlib import pyplot
from scipy.stats import normaltest

results=DataFrame()
results['Model3'] = read_csv('results1.csv', header=None).values[:, 0]
results['Model2'] = read_csv('results2.csv', header=None).values[:, 0]

result1 = read_csv('results1.csv', header=None)
value, p = normaltest(result1.values[:,0])
print(value, p)
if p >= 0.05:
	print('It is likely that result1 is normal')
else:
	print('It is unlikely that result1 is normal')

result2 = read_csv('results2.csv', header=None)
value, p = normaltest(result2.values[:,0])
print(value, p)
if p >= 0.05:
	print('It is likely that result2 is normal')
else:
	print('It is unlikely that result2 is normal')

# descriptive stats
print(results.describe())
# box and whisker plot
results.boxplot()
pyplot.show()
# histogram
results.hist()
pyplot.show()

values1 = results['Model2']
values2 = results['Model3']

# calculate the significance
value, pvalue = ttest_ind(values1, values2, equal_var=True)
print(value, pvalue)
if pvalue > 0.05:
	print('Samples are likely drawn from the same distributions (fail to reject H0)')
else:
	print('Samples are likely drawn from different distributions (reject H0)')
