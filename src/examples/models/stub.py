import radixdb.executor
import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt

df = radixdb.executor.df_query("select date::date, data from co2;", "postgres://postgres@localhost:5432/radixdb")
exec(open("./co2_model.py").read(), {"_df": df})
plt.savefig('co2.png')
