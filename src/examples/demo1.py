import radixdb.evaluator
c=radixdb.evaluator.eval("Q(T.hist_prices).fields(['high', 'low']).limit(10).show()")
print(c)
