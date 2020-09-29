#streamlit run demo5.py
import psycopg2
import os
import pandas as pd
import streamlit as st

# @st.cache prevents the streamlit app from executing redundant processes
# repeatedly that are expensive such as database connections or reading input
@st.cache(allow_output_mutation=True)
def get_query_results():
    """ A function that returns a table of members.
        Until streamlit provides a way to pageinate results,
        maximum of 1000 rows will be returned.
        Output can be either a streamlit rendered dataframe
        or raw HTML version.  Raw HTML version will provide
        a hyperlink that will take you directly to the person's
        company profile page.  This can be used to double-check
        that the profile URL has been correctly generated.
    """

    # Connect to the PostgreSQL database server
    with psycopg2.connect(host='localhost',
                          port='5432',
                          database='radixdb',
                          user='postgres',
                          password=None) as conn:

        sql = """
         select * from pg_stat_statements order by total_time;
        """

        # Execute query and return results as a pandas dataframe
        df = pd.read_sql(sql, conn, index_col=None)
        # Define a function to create a "Profile" hyperlink
        def createProfileHref(url: str):
            """ Function to create a new column that converts URL as HTML hyperlink """
    
            value = '<a href="' + str(url) + '"' + "/>Profile</a>"
    
            return value

        # Apply the function we created above and create our new hyperlink column
        df['profile_href'] = df['userid'].apply(createProfileHref)

        # Change order of dataframe columns
        df = df[['userid', 'dbid', 'queryid', 'total_time', 
                 'query']]

    return df

def write():
    """ Writes content to the app """
    st.title("Get execution stats From RadixDB")
    # Check to see if checkbox was checked or not (boolean) and will be used to
    # determine if the output should be a streamlit dataframe or raw HTML.
    html = st.checkbox(
        'OPTIONAL: Render output as raw html to access the \"Profile\" hyperlink. ' +
         'Otherwise, just click on Execute botton.',
        False)

    # Define what happens when user clicks on the "Execute" button
    if st.button("Execute"):
        '''
        ### Query results:
        '''
        if html:
            # Render or display the raw HTML version of the dataframe
            st.write(get_query_results().to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            # Render or display the streamlit dataframe
            st.dataframe(get_query_results())

if __name__ == "__main__":
    write()            
