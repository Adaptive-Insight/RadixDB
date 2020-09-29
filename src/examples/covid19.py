"""
  git clone https://github.com/datasets/covid-19.git
"""
import pathlib
from radixdb.executor import run_sql

p=str(pathlib.Path().absolute())
print("loading data from ", p)
run_sql('postgresql://postgres@localhost/radixdb', """
drop table if exists us_deaths;
create table us_deaths(
Lat numeric, Population int, Date date, cases int ,Long numeric, Country text,State text);""")
run_sql('postgresql://postgres@localhost/radixdb',
"""COPY us_deaths(Lat,Population, Date, cases,Long,Country, State)
FROM '""" + p + "/us_deaths.csv' DELIMITER ',' CSV HEADER;""")

run_sql('postgresql://postgres@localhost/radixdb', """
drop table if exists us_confirmed;
create table us_confirmed(
Lat numeric, Population int, Date date, cases int ,Long numeric, Country text,State text);""")
run_sql('postgresql://postgres@localhost/radixdb',
"""COPY us_confirmed(Lat,Population, Date, cases,Long,Country, State)
FROM '""" + p + """/us_deaths.csv' DELIMITER ',' CSV HEADER;""")

run_sql('postgresql://postgres@localhost/radixdb', """
drop table if exists covid19_worldwide;
create table covid19_worldwide(
Date date,Country text,State text,Lat numeric,Long numeric,Confirmed int,Recovered int,Deaths int);""")
run_sql('postgresql://postgres@localhost/radixdb',
"""COPY covid19_worldwide(Date,Country,State,Lat,Long,Confirmed,Recovered,Deaths)
FROM '""" + p + """/time-series-19-covid-combined.csv' DELIMITER ',' CSV HEADER;""")
