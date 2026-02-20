import streamlit as st 
import pandas as pd 

import utils

import preprocessing.missing as missing
import preprocessing.outlier as outlier
import preprocessing.feature_construction as contruct
import preprocessing.encoding as encoding
import preprocessing.scaling as scale
import preprocessing.feature_selection as selection
import preprocessing.transformation as transformation
import preprocessing.datetime as dtime
import preprocessing.duplicate as duplicate

st.set_page_config(layout="wide")

st.title("Data Preprocessor")

if "df" not in st.session_state:
    st.session_state["df"] = None

file = st.file_uploader("Upload your dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.session_state["df"] = df

if st.session_state["df"] is not None:

    df = st.session_state["df"]

    ### SIDEBAR ###

    option = st.sidebar.selectbox(
        "Choose Operation :",
        ["Handle Missing Values", "Handle Outlier", "Feature Construction", "Encoding", "Scaling", "Feature Selection"],
        index=None,
        placeholder="Select Option"
    )

    ### Basic INFO ###

    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Shape of Dataset", f"{utils.shape(st.session_state.df)[0]} x {utils.shape(st.session_state.df)[1]}")
    col2.metric("Duplicate Values", utils.duplicate_val(st.session_state.df))
    col3.metric("Dataset Empty", utils.check_empty(st.session_state.df))
    col4.write("Column Datatypes")
    col4.write(utils.column_dtypes(st.session_state.df))

    st.markdown("### Sample of Dataset")
    st.dataframe(utils.struct(st.session_state.df))

    col6, col7, col8, col9 = st.columns(4)

    with col6:
        st.markdown("### Numeric Columns")
        st.write(utils.get_numeric_columns(st.session_state.df).columns)
    
    with col7:
        st.markdown("### Catgorical Columns")
        st.write(utils.get_categorical_columns(st.session_state.df).columns)
    
    with col8:
        st.markdown("### Boolean Columns")
        st.write(utils.get_bool_columns(st.session_state.df).columns)

    with col9:
        st.markdown("### DateTime Columns")
        st.write(utils.get_datetime_columns(st.session_state.df).columns)

    col10, col11 = st.columns(2)

    with col10:
        st.markdown("### Missing value")
        st.dataframe(utils.missing_val(st.session_state.df))

    with col11:
        st.markdown("### Missing value %")
        st.dataframe(utils.missing_value_per(st.session_state.df))

    st.markdown("### Description")
    st.dataframe(utils.description(st.session_state.df))

    st.markdown("### Correlation")
    st.dataframe(utils.correlation(st.session_state.df))

    ### HANDLING MISSING VALUES ###

    if option == "Handle Missing Values":

        num_missing = utils.missing_val(utils.get_numeric_columns(st.session_state.df))
        cat_missing = utils.missing_val(utils.get_categorical_columns(st.session_state.df))

        if (num_missing == 0).all() and (cat_missing == 0).all():
            st.success("No missing value")
        
        else:
            st.markdown("### Choose Column for Imputation")

            column_type = st.selectbox("Choose type of Column:", ["Numerical", "Categorical"], index=None, placeholder="Select Option")

            if column_type == "Numerical":

                if (num_missing == 0).all():
                    st.success("No missing value in Numerical Column")

                else:
                    num_col = utils.get_numeric_columns(st.session_state.df)
                    col_mis_num = st.session_state.df[
                        [col for col in num_col.columns if num_col[col].isnull().sum() > 0]
                    ]
                    num_option = st.selectbox("Numerical Column", ["All Numerical Columns"] + list(col_mis_num), index=None, placeholder="Select Option")
                    
                    if num_option == "All Numerical Columns":

                        st.markdown("### Choose type of Imputation")
                        imp = st.selectbox("Operations:", ["Mean", "Median", "Random", "KNN Imputation"], index=None, placeholder="Select Option")

                        if imp == "Mean":
                            col_mis_num = missing.mean_imputation_all_feature(col_mis_num)
                            st.session_state.df.update(col_mis_num)
                            st.success(f"{num_option} imputed successfully")

                        elif imp == "Median":
                            col_mis_num = missing.median_imputation_all_feature(col_mis_num)
                            st.session_state.df.update(col_mis_num)
                            st.success(f"{num_option} imputed successfully")

                        elif imp == "Random":
                            col_mis_num = missing.random_imputation_all_feature(col_mis_num)
                            st.session_state.df.update(col_mis_num)
                            st.success(f"{num_option} imputed successfully")

                        elif imp == "KNN Imputation":
                            n = st.number_input("Input no. of neighbour: ", min_value=0, max_value=10, step=1)
                            weight = st.selectbox("Weight", ["uniform", "distance"], index=None, placeholder="Select Option")
                            
                            if weight != None:

                                col_mis_num = missing.knn_imputer(col_mis_num, n , weight)
                                st.session_state.df.update(col_mis_num)
                                st.success(f"{num_option} imputed successfully")

                    elif num_option != None:
                        st.markdown("### Choose type of Imputation")
                        imp = st.selectbox("Operations:", ["Mean", "Median", "Random"], index=None, placeholder="Select Option")

                        if imp == "Mean":
                            st.session_state.df[num_option] = missing.mean_imputation_single_feature(num_option)
                            st.success(f"{num_option} imputed successfully")

                        elif imp == "Median":
                            st.session_state.df[num_option] = missing.median_imputation_single_feature(num_option)
                            st.success(f"{num_option} imputed successfully")

                        elif imp == "Random":
                            st.session_state.df = missing.random_imputation_single_feature(st.session_state.df, num_option)
                            st.success(f"{num_option} imputed successfully")
            
            if column_type == "Categorical":

                if (cat_missing == 0).all():
                    st.success("No missing value in Categorical Column")
                
                else:
                    cat_col = utils.get_categorical_columns(df)
                    col_mis_cat = [col for col in cat_col.columns if cat_col[col].isnull().sum() > 0]
                    cat_option = st.selectbox("Categorical Column", ["All Categorical Columns"] + list(col_mis_cat), index=None, placeholder="Select Option")
