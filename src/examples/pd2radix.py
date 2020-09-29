import pandas as pd
df = pd.DataFrame({
    'name':['john','mary','peter','jeff','bill','lisa','jose'],
    'age':[23,78,22,19,45,33,20],
    'gender':['M','F','M','M','M','F','M'],
    'state':['california','dc','california','dc','california','texas','texas'],
    'num_children':[2,0,0,3,2,1,4],
    'num_pets':[5,1,0,5,2,2,3]
})

import radixdb.executor
radixdb.executor.df_create(df, "fam", "postgres://postgres@localhost:5432/radixdb");
radixdb.executor.df_copy(df, "fam", "postgres://postgres@localhost:5432/radixdb");

import radixdb.executor
radixdb.executor.df_create(df, "co2", "postgres://postgres@localhost:5432/radixdb");
