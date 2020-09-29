import numpy as np
import pandas as pd
import radixdb.executor

np.save('co2_by_month', co2_by_month)
co2_by_month = np.load('./co2_by_month.npy')
df = pd.DataFrame({'data': co2_by_month})


np.save('demand', demand)
demand = np.load('./demand.npy')

np.save('temperature', temperature)
temperature = np.load('./temperature.npy')

co2_by_month = np.load('./co2_by_month.npy')
df = pd.DataFrame({'date': np.arange("1966-01", "2019-02", dtype="datetime64[M]"), 'data': co2_by_month})
radixdb.executor.df_create(df, "co2", "postgres://postgres@localhost:5432/radixdb");
radixdb.executor.df_copy(df, "co2", "postgres://postgres@localhost:5432/radixdb");

demand = np.load('./demand.npy')
temperature = np.load('./temperature.npy')
df = pd.DataFrame({'date':np.arange('2014-01-01', '2014-02-26', dtype='datetime64[h]'), 'demand': demand, 'temperature': temperature })
radixdb.executor.df_create(df, "electricity", "postgres://postgres@localhost:5432/radixdb");
radixdb.executor.df_copy(df, "electricity", "postgres://postgres@localhost:5432/radixdb");
