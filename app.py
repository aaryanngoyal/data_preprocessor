import streamlit as st 
import pandas as pd 

import utils

import preprocessing.missing as missing
import preprocessing.outlier as outlier
import preprocessing.feature_construction as construct
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
    
if "imputation_message" not in st.session_state:
    st.session_state.imputation_message = ""

if "outlier_message" not in st.session_state:
    st.session_state.outlier_message = ""
    
if "construct_message" not in st.session_state:
    st.session_state.construct_message = ""

if "encoding_message" not in st.session_state:
    st.session_state.encoding_message = ""

if "scaling_message" not in st.session_state:
    st.session_state.scaling_message = ""

if "select_message" not in st.session_state:
    st.session_state.select_message = ""

if "widget_key" not in st.session_state:
    st.session_state.widget_key = 0

file = st.file_uploader("Upload your dataset", type=["csv"])

if file and st.session_state["df"] is None:
    st.session_state["df"] = pd.read_csv(file)

if st.session_state["df"] is not None:

    df = st.session_state["df"]

    ### SIDEBAR ###

    option = st.sidebar.selectbox(
        "Choose Operation :",
        ["Handle Missing Values", "Handle Outlier", "Feature Construction", "Encoding", "Scaling", "Feature Selection", "Export File"],
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

    drop = st.sidebar.checkbox("Drop Duplicates")
    if drop and utils.duplicate_val(st.session_state.df) > 0:
        st.session_state.df = st.session_state.df.drop_duplicates()
        st.rerun()

    ### HANDLING MISSING VALUES ###

    if option == "Handle Missing Values":

        num_missing = utils.missing_val(utils.get_numeric_columns(st.session_state.df))
        cat_missing = utils.missing_val(utils.get_categorical_columns(st.session_state.df))

        if st.session_state.imputation_message:
            st.success(st.session_state.imputation_message)
            st.session_state.imputation_message = ""

        if (num_missing == 0).all() and (cat_missing == 0).all():
            st.success("No missing value")
        
        else:
            
            st.markdown("### Choose Column for Imputation")

            column_type = st.selectbox("Choose type of Column:", ["Numerical", "Categorical"], key=f"col_type_{st.session_state.widget_key}", index=None, placeholder="Select Option")

            if column_type == "Numerical":
                
                num_missing_live = utils.missing_val(utils.get_numeric_columns(st.session_state.df))
                
                if (num_missing_live == 0).all():

                    st.success("No missing value in Numerical Column")

                else:

                    num_col = utils.get_numeric_columns(st.session_state.df)
                    col_mis_num = st.session_state.df[
                        [col for col in num_col.columns if st.session_state.df[col].isnull().sum() > 0]
                    ]

                    num_option = st.selectbox("Numerical Column", ["All Numerical Columns"] + list(col_mis_num), key=f"num_opt_{st.session_state.widget_key}", index=None, placeholder="Select Option")
                    
                    if num_option == "All Numerical Columns":

                        st.markdown("### Choose type of Imputation")
                        imp = st.selectbox("Operations:", ["Mean", "Median", "Random", "KNN Imputation"], key=f"imp_{st.session_state.widget_key}", index=None, placeholder="Select Option")

                        # MEAN IMPUTATION 

                        if imp == "Mean":
                            if st.button("Apply Mean Imputation"):
                                col_mis_num = missing.mean_imputation_all_feature(col_mis_num)
                                for col in col_mis_num.columns:
                                    st.session_state.df[col] = col_mis_num[col]
                                st.session_state.imputation_message = f"{num_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()
                        
                        # MEDIAN IMPUTATION 

                        elif imp == "Median":
                            if st.button("Apply Median Imputation"):
                                col_mis_num = missing.median_imputation_all_feature(col_mis_num)
                                for col in col_mis_num.columns:
                                    st.session_state.df[col] = col_mis_num[col]
                                st.session_state.imputation_message = f"{num_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()

                        # RANDOM IMPUTATION 

                        elif imp == "Random":
                            if st.button("Apply Random Imputation"):
                                col_mis_num = missing.random_imputation_all_feature(col_mis_num)
                                for col in col_mis_num.columns:
                                    st.session_state.df[col] = col_mis_num[col]
                                st.session_state.imputation_message = f"{num_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()

                        # KNN IMPUTATION 

                        elif imp == "KNN Imputation":
                            if st.button("Apply KNN Imputation"):
                                n = st.number_input("Input no. of neighbour: ", min_value=0, max_value=10, step=1)
                                weight = st.selectbox("Weight", ["uniform", "distance"], index=None, placeholder="Select Option")
                                
                                if weight is not None:

                                    col_mis_num = missing.knn_imputer(col_mis_num, n , weight)
                                    for col in col_mis_num.columns:
                                        st.session_state.df[col] = col_mis_num[col]
                                    st.session_state.imputation_message = f"{num_option} imputed successfully"
                                    st.session_state.widget_key += 1
                                    st.rerun()

                    elif num_option is not None:

                        st.markdown("### Choose type of Imputation")
                        imp = st.selectbox("Operations:", ["Mean", "Median", "Random"], key=f"imp_{st.session_state.widget_key}", index=None, placeholder="Select Option")

                        # MEAN IMPUTATION 

                        if imp == "Mean":
                            if st.button("Apply Mean Imputation"):
                                st.session_state.df = missing.mean_imputation_single_feature(st.session_state.df, num_option)
                                st.session_state.imputation_message = f"{num_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()

                        # MEDIAN IMPUTATION 

                        elif imp == "Median":
                            if st.button("Apply Median Imputation"):
                                st.session_state.df = missing.median_imputation_single_feature(st.session_state.df, num_option)
                                st.session_state.imputation_message = f"{num_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()

                        # RANDOM IMPUTATION 

                        elif imp == "Random":
                            if st.button("Apply Random Imputation"):
                                st.session_state.df = missing.random_imputation_single_feature(st.session_state.df, num_option)
                                st.session_state.imputation_message = f"{num_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()

            elif column_type == "Categorical":

                cat_missing_live = utils.missing_val(utils.get_categorical_columns(st.session_state.df))
                
                if (cat_missing_live == 0).all():
                    st.success("No missing value in Categorical Column")
                
                else:

                    cat_col = utils.get_categorical_columns(st.session_state.df)
                    col_mis_cat = st.session_state.df[
                        [col for col in cat_col.columns if st.session_state.df[col].isnull().sum() > 0]
                    ]
                    cat_option = st.selectbox("Categorical Column", ["All Categorical Columns"] + list(col_mis_cat), key=f"cat_opt_{st.session_state.widget_key}", index=None, placeholder="Select Option")

                    if cat_option == "All Categorical Columns":

                        st.markdown("### Choose type of Imputation")
                        imp = st.selectbox("Operations:", ["Mode", "Add Missing Value"], key=f"imp_{st.session_state.widget_key}", index=None, placeholder="Select Option")

                        # MODE IMPUTATION 

                        if imp == "Mode":
                            if st.button("Apply Mode Imputation"):
                                col_mis_cat = missing.mode_imputation_all_features(col_mis_cat)
                                for col in col_mis_cat.columns:
                                    st.session_state.df[col] = col_mis_cat[col]
                                st.session_state.imputation_message = f"{cat_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()

                        # ADD MISSING VALUE IMPUTATION 

                        elif imp == "Add Missing Value":
                            if st.button("Apply Missing Value Imputation"):
                                col_mis_cat = missing.add_missing_value_all_feature(col_mis_cat)
                                for col in col_mis_cat.columns:
                                    st.session_state.df[col] = col_mis_cat[col]
                                st.session_state.imputation_message = f"{cat_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()

                    elif cat_option is not None:
                        st.markdown("### Choose type of Imputation")
                        imp = st.selectbox("Operations:", ["Mode", "Add Missing Value"], key=f"imp_{st.session_state.widget_key}", index=None, placeholder="Select Option")

                        # MODE IMPUTATION 

                        if imp == "Mode":
                            if st.button("Apply Mode Imputation"):
                                st.session_state.df = missing.mode_imputation_single_features(st.session_state.df, cat_option)
                                st.session_state.imputation_message = f"{cat_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()

                        # ADD MISSING VALUE IMPUTATION 

                        elif imp == "Add Missing Value":
                            if st.button("Apply Missing Value Imputation"):
                                st.session_state.df = missing.add_missing_value_single_feature(st.session_state.df, cat_option)
                                st.session_state.imputation_message = f"{cat_option} imputed successfully"
                                st.session_state.widget_key += 1
                                st.rerun()

    # HANDLING OUTLIER

    elif option == "Handle Outlier":

        if st.session_state.outlier_message:
            st.success(st.session_state.outlier_message)
            st.session_state.outlier_message = ""

        if (utils.missing_val(st.session_state.df)).any() > 0:
            st.error("Impute Missing Value")

        else:
            st.markdown("### Outlier Detection")

            out_option = st.selectbox("Select Method for Outlier Detection", ["ZScore", "IQR", "Percentile"], key=f"out_opt{st.session_state.widget_key}", index=None, placeholder="Select Option")
            num_col = utils.get_numeric_columns(st.session_state.df)

            if out_option is not None:
                out_col = st.selectbox("Choose Column", list(num_col.columns), key=f"out_col{st.session_state.widget_key}", index=None, placeholder="Select Option")
                
                if out_col is not None:

                    # Z SCORE 

                    if out_option == "ZScore":
                        if num_col[out_col].skew() >= -0.5 and num_col[out_col].skew() <= 0.5:
                            out_z = outlier.finding_outlier_z_score(st.session_state.df, out_col)
                            st.dataframe(out_z)

                            if len(out_z) > 0:
                                st.markdown("### Outlier Removal")
                                out = st.selectbox("Choose Technique", ["Trimming", "Capping"], key=f"out_{st.session_state.widget_key}", index=None, placeholder="Select Options")

                                # Z SCORE TRIMMNG

                                if out == "Trimming":
                                    if st.button("Apply Trimming"):
                                        st.session_state.df = outlier.outlier_removal_zscore_trimming(st.session_state.df, out_col)
                                        st.session_state.outlier_message = f"Outliers handeled in {out_col} using Z Score - Trimming"
                                        st.session_state.widget_key += 1
                                        st.rerun()

                                # Z SCORE CAPPING

                                elif out == "Capping":
                                    if st.button("Apply Capping"):
                                        st.session_state.df = outlier.outlier_removal_zscore_capping(st.session_state.df, out_col)
                                        st.session_state.outlier_message = f"Outliers handeled in {out_col} using Z Score - Capping"
                                        st.session_state.widget_key += 1
                                        st.rerun()
                                        
                            else:
                                st.success("No Outliers Present")
                        
                        else:
                            st.error(f"{out_col} not normalized, Consider another way for detection and removal of outlier")

                    # IQR

                    elif out_option == "IQR":
                        if num_col[out_col].skew() < -0.5 or num_col[out_col].skew() > 0.5:
                            out_iqr = outlier.finding_outlier_iqr(st.session_state.df, out_col)
                            st.dataframe(out_iqr)

                            if len(out_iqr) > 0:
                                st.markdown("### Outlier Removal")
                                out = st.selectbox("Choose Technique", ["Trimming", "Capping"], key=f"out_{st.session_state.widget_key}", index=None, placeholder="Select Options")

                                # IQR TRIMMNG

                                if out == "Trimming":
                                    if st.button("Apply Trimming"):
                                        st.session_state.df = outlier.outlier_removal_iqr_trimming(st.session_state.df, out_col)
                                        st.session_state.outlier_message = f"Outliers handeled in {out_col} using IQR - Trimming"
                                        st.session_state.widget_key += 1
                                        st.rerun()

                                # IQR CAPPING

                                elif out == "Capping":
                                    if st.button("Apply Capping"):
                                        st.session_state.df = outlier.outlier_removal_iqr_capping(st.session_state.df, out_col)
                                        st.session_state.outlier_message = f"Outliers handeled in {out_col} using IQR - Capping"
                                        st.session_state.widget_key += 1
                                        st.rerun()

                            else:
                                st.success("No Outliers Present")

                        else:
                            st.error(f"{out_col} normalized, Consider applying Z Score")

                    # PERCENTILE

                    elif out_option == "Percentile":
                        if st.session_state.df[out_col].skew() < -0.5 or st.session_state.df[out_col].skew() > 0.5:
                            out_percentile = outlier.finding_outlier_percentile(st.session_state.df, out_col)
                            st.dataframe(out_percentile)

                            if len(out_percentile) > 0:
                                st.markdown("### Outlier Removal")
                                out = st.selectbox("Choose Technique", ["Trimming", "Capping"], key=f"out_{st.session_state.widget_key}", index=None, placeholder="Select Options")

                                # PERCENTILE TRIMMNG

                                if out == "Trimming":
                                    if st.button("Apply Trimming"):
                                        st.session_state.df = outlier.outlier_removal_percentile_trimming(st.session_state.df, out_col)
                                        st.session_state.outlier_message = f"Outliers handeled in {out_col} using Percentile - Trimming"
                                        st.session_state.widget_key += 1
                                        st.rerun()

                                # PERCENTILE CAPPING

                                elif out == "Capping":
                                    if st.button("Apply Capping"):
                                        st.session_state.df = outlier.outlier_removal_percentile_capping(st.session_state.df, out_col)
                                        st.session_state.outlier_message = f"Outliers handeled in {out_col} using Percentile - Capping"
                                        st.session_state.widget_key += 1
                                        st.rerun()

                            else:
                                st.success("No Outliers Present")

                        else:
                            st.error(f"{out_col} normalized, Consider applying Z Score")

    # FEATURE CONSTRUCTION

    elif option == "Feature Construction":

        st.markdown("### Feature Construction")
        feat_option = st.selectbox("Choose Type of Feature Construction", ["Convert to Bin", "Polynomial Feature", "Ratio of Features", "Difference in Features", "Addition of Features", "Multiplication of Features"],   key=f"opt_{st.session_state.widget_key}", index=None, placeholder="Select Option")
        
        if st.session_state.construct_message:
            st.success(st.session_state.construct_message)
            st.session_state.construct_message = ""

        # CONSTRUCT USING BINS

        if feat_option == "Convert to Bin":
            
            bins = st.number_input("Enter no. of bins", min_value=2, max_value=20, step=1)
            encoding = st.selectbox("Enter type of Encoding", ["onehot", "onehot-dense", "ordinal"],  key=f"enc_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            
            if encoding is not None:    
                column = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)),  key=f"col_{st.session_state.widget_key}", index=None, placeholder="Select Option")
                
                if column is not None:
                    strat = st.selectbox("Enter type of Strategy", ["uniform", "quantile", "kmeans"],  key=f"strat_{st.session_state.widget_key}", index=None, placeholder="Select Option")
                    
                    if strat is not None:
                        
                        if st.button("Apply Binning"):
                            try:
                                st.session_state.df = construct.bin_numeric(st.session_state.df, column, bins, strat, encoding)
                                st.session_state.construct_message = "Feature has been constructed using Bins"
                                st.session_state.widget_key += 1
                                st.rerun()
                            except ValueError as e:
                                st.error(str(e))
                    
        # CONSTRUCT USING POLYNOMIAL FEATURE

        elif feat_option == "Polynomial Feature":
            
            col = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), index=None, placeholder="Select Option")
            deg = st.number_input("Enter Degree", min_value=2, max_value=20, step=0)
            
            if col is not None:
                if st.button("Apply Polynomial Feature"):
            
                    st.session_state.df = construct.poly_feature(st.session_state.df, col, deg)
                    st.session_state.construct_message = "Feature has been constructed using Bins"
                    st.session_state.widget_key += 1
                    st.rerun()

        # CONSTRUCT USING RATIO OF FEATURE

        elif feat_option == "Ratio of Features":

            col1 = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col1_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            col2 = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col2_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            
            if col1 is not None and col2 is not None:
                if col1 == col2:
                    st.error("Don't select same column")
                else:
                    if st.button("Apply Ratio"):
                        st.session_state.df = construct.ratio_feature(st.session_state.df, col1, col2)
                        st.session_state.construct_message = "Feature has been constructed using Bins"
                        st.session_state.widget_key += 1
                        st.rerun()

        # CONSTRUCT USING DIFFERENCE IN FEATURE

        elif feat_option == "Difference in Features":
            
            col1 = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col1_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            col2 = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col2_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            
            if col1 is not None and col2 is not None:
                if col1 == col2:
                    st.error("Don't select same column")
                else:
                    if st.button("Apply Difference"):
                        st.session_state.df = construct.difference_feature(st.session_state.df, col1, col2)
                        st.session_state.construct_message = "Feature has been constructed using Bins"
                        st.session_state.widget_key += 1
                        st.rerun()

        # CONSTRUCT USING ADDITION OF FEATURE

        elif feat_option == "Addition of Features":

            col1 = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col1_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            col2 = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col2_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            
            if col1 is not None and col2 is not None:
                if col1 == col2:
                    st.error("Don't select same column")
                else:
                    if st.button("Apply Addition"):
                        st.session_state.df = construct.add_feature(st.session_state.df, col1, col2)
                        st.session_state.construct_message = "Feature has been constructed using Bins"
                        st.session_state.widget_key += 1
                        st.rerun()

        # CONSTRUCT USING MULTIPLICATION OF FEATURE

        elif feat_option == "Multiplication of Features":
            
            col1 = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col1_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            col2 = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col2_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            
            if col1 is not None and col2 is not None:
                if col1 == col2:
                    st.error("Don't select same column")
                else:
                    if st.button("Apply Multiplication"):
                        st.session_state.df = construct.multiply_feature(st.session_state.df, col1, col2)
                        st.session_state.construct_message = "Feature has been constructed using Bins"
                        st.session_state.widget_key += 1
                        st.rerun()

    # ENCODING

    elif option == "Encoding":
        
        st.markdown("### Encoding")

        encode_option = st.selectbox("Select type of Encoding", ["One-Hot Encoding", "Label Encoding", "Ordinal Encoding"], key=f"opt_{st.session_state.widget_key}", index=None, placeholder="Select Option")

        if st.session_state.encoding_message:
            st.success(st.session_state.encoding_message)
            st.session_state.encoding_message = ""

        # ONE HOT ENCODING

        if encode_option == "One-Hot Encoding":
            col = st.selectbox("Select Column to Encode", list(utils.get_categorical_columns(st.session_state.df)), key=f"col_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            if col is not None:
                if st.button("Apply One Hot Encoding"):
                    try:
                        st.session_state.df = encoding.one_hot_encode(st.session_state.df, col)
                        st.session_state.encoding_message = "Feature has been encoded using one hot encoding"
                        st.session_state.widget_key += 1
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        # LABEL ENCODING

        elif encode_option == "Label Encoding":
            col = st.selectbox("Select Column to Encode", list(utils.get_categorical_columns(st.session_state.df)), key=f"col_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            if col is not None:
                if st.button("Apply Label Encoding"):
                    st.session_state.df = encoding.label_encoding(st.session_state.df, col)
                    st.session_state.encoding_message = "Feature has been encoded using label encoding"
                    st.session_state.widget_key += 1
                    st.rerun()

        # ORDINAL ENCODING

        elif encode_option == "Ordinal Encoding":
            col = st.selectbox("Select Column to Encode", list(utils.get_categorical_columns(st.session_state.df)), key=f"col_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            if col is not None:
                unique_val = st.session_state.df[col].dropna().unique().tolist()
                st.write("Define order (low to high)", unique_val)
                general = st.multiselect("Order Categories", unique_val, key=f"ord_{st.session_state.widget_key}")
                if len(general) < len(unique_val):
                    st.warning(f"Select all {len(unique_val)} categories to proceed. ({len(general)}/{len(unique_val)} selected)")
                else:
                    if st.button("Apply Ordinal Encoding"):
                        st.session_state.df = encoding.ordinal_encoding(st.session_state.df, col, general)
                        st.session_state.encoding_message = "Feature has been encoded using ordinal encoding"
                        st.session_state.widget_key += 1
                        st.rerun()

    # SCALING   

    elif option == "Scaling":

        st.markdown("### Scaling")

        scale_option = st.selectbox("Select type of Scaling", ["Standardization", "Min Max Scale", "Max Absolute Scale", "Robust Scale"], key=f"sca_{st.session_state.widget_key}", index=None, placeholder="Select Option")

        if st.session_state.scaling_message:
            st.success(st.session_state.scaling_message)
            st.session_state.scaling_message = ""

        # STANDARDIZATION

        if scale_option == "Standardization":
            col = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            if col is not None:
                mean = st.selectbox("with_mean", [True, False], key=f"mean_{st.session_state.widget_key}", index=None, placeholder="Select Option")
                std = st.selectbox("with_std", [True, False], key=f"std_{st.session_state.widget_key}", index=None, placeholder="Select Option")

                if mean is not None and std is not None:
                    if st.button("Apply Standarization"):
                        st.session_state.df = scale.standardization(st.session_state.df, col, mean, std)
                        st.session_state.scaling_message = "Feature has been scaled using Standardization"
                        st.session_state.widget_key += 1
                        st.rerun()

        # MIN MAX SCALE

        elif scale_option == "Min Max Scale":
            col = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            if col is not None:
                if st.button("Apply Min Max Scale"):
                    st.session_state.df = scale.min_max_scale(st.session_state.df, col)
                    st.session_state.scaling_message = "Feature has been scaled using Min Max Scale"
                    st.session_state.widget_key += 1
                    st.rerun()

        # MAX ABSOLUTE SCALE

        elif scale_option == "Max Absolute Scale":
            col = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            if col is not None:
                if st.button("Apply Max Absolute Scale"):
                    st.session_state.df = scale.max_abs_scale(st.session_state.df,col)
                    st.session_state.scaling_message = "Feature has been scaled using Max Absolute Scale"
                    st.session_state.widget_key += 1
                    st.rerun()

        # ROBUST SCALE

        elif scale_option == "Robust Scale":
            col = st.selectbox("Select Column", list(utils.get_numeric_columns(st.session_state.df)), key=f"col_{st.session_state.widget_key}", index=None, placeholder="Select Option")
            if col is not None:
                if st.button("Apply Robust Scale"):
                    st.session_state.df = scale.robust_scale(st.session_state.df, col)
                    st.session_state.scaling_message = "Feature has been scaled using Robust Scale"
                    st.session_state.widget_key += 1
                    st.rerun()
    
    # FEATURE SELECTION

    elif option == "Feature Selection":
        
        st.markdown("### Feature Selection")

        select_option = st.selectbox("Select type of Feature Selection", ["Remove Constant Column", "Remove High Correlation"], key=f"opt_{st.session_state.widget_key}", index=None, placeholder="Select Option")

        if st.session_state.select_message:
            st.success(st.session_state.select_message)
            st.session_state.select_message = ""

        # REMOVE CONSTANT COLUMN

        if select_option == "Remove Constant Column":
            if st.button("Remove Constant Column"):
                st.session_state.df = selection.remove_constant_constant_column(st.session_state.df)
                st.session_state.select_message = "Constant Column has been removed"
                st.session_state.widget_key += 1
                st.rerun()

        # REMOVE HIGH CORRELATION

        elif select_option == "Remove High Correlation":
            threshold = st.slider("Select Threshold", 0.0 , 1.0, 0.8, key=f"thresh_{st.session_state.widget_key}")
            if st.button("Remove High Correlation"):
                st.session_state.df = selection.remove_high_correlation(st.session_state.df, threshold)
                st.session_state.select_message = "High Correlation has been removed"
                st.session_state.widget_key += 1
                st.rerun()

    # EXPORT FILE

    elif option == "Export File":
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="Download Processed CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )