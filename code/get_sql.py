"""Connects to MySQL database, executes query and returns it as a Pandas df"""

import sqlalchemy
import pandas as pd
import os

def get_df(query):
    cnx = connect_to_db()
    df = pd.read_sql(query, cnx)
    return df

def connect_to_db():
    engine = sqlalchemy.create_engine("mysql+mysqldb://{}:{}@{}/capublic" \
        .format(os.environ["MYSQL_USER"], os.environ["MYSQL_PWD"], \
        os.environ["MYSQL_HOST"]))
    cnx = engine.raw_connection()
    return cnx
