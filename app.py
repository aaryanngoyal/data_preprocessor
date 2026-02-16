import streamlit as st 
import pandas as pd 
import utils

st.title("Upload Dataset")

if "df" not in st.session_state:
    st.session_state["df"] = None

file = st.file_uploader("Upload your dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.session_state["df"] = df

if st.session_state["df"] is not None:

    df = st.session_state["df"]

    ### SIDEBAR ###

    st.sidebar.markdown("### Shape of DateSet  ")
    st.sidebar.write(utils.shape(df))

    st.sidebar.markdown("### Information of Dataset")
    st.sidebar.text(utils.df_info(df))

    st.sidebar.markdown("### Duplicate Values")
    st.sidebar.write(utils.duplicate_val(df))

    st.sidebar.markdown("### Dataset Empty")
    st.sidebar.write(utils.check_empty(df))

    st.sidebar.markdown("### Column DataType")
    st.sidebar.table(utils.column_dtypes(df))

    ### UNDERSTANDING DATASET ###

    st.markdown("### Sample of Dataset")
    st.dataframe(utils.struct(df))

    st.markdown("### Missing Values")
    st.write(utils.missing_val(df))

    st.markdown("### Description")
    st.dataframe(utils.description(df))

    st.markdown("### Correlation")
    st.dataframe(utils.correlation(df))

    st.markdown("### Numeric Columns")
    st.write(utils.get_numeric_columns(df).columns)

    st.markdown("### Catgorical Columns")
    st.write(utils.get_categorical_columns(df).columns)

    st.markdown("### Boolean Columns")
    st.write(utils.get_bool_columns(df).columns)

    st.markdown("### Missing value %")
    st.dataframe(utils.missing_value_per(df))

    ### HANDLING MISSING VALUES ###

    st.markdown("# HANDLING MISSING VALUES")

    st.sidebar.markdown("### Choose Column for Imputation")
    
    st.sidebar.markdown("### Numeric Column")
    st.sidebar.selectbox("Features", list(utils.get_numeric_columns(df).columns) + ["Entire Dataset"])

    st.sidebar.markdown("### Categorical Column")
    st.sidebar.selectbox("Features", list(utils.get_categorical_columns(df).columns) + ["Entire Dataset"])
