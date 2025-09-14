import streamlit as st
import pandas as pd
import io
from Data_Cleaning import clean_dataframe
from Fuzzy_Matching import exact_match, fuzzy_match_blocking,build_final_output



st.set_page_config(page_title="Text Matching App 🔍", layout="wide")

# Title
st.title("Text Matching App 🔍")

# Upload Section
uploaded_file = st.file_uploader("**Upload your Excel file**: containing only the two columns to match.", type=["xlsx", "xls"])



# Preview uploaded file
if uploaded_file:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)

        # Validation: check column count
        if df.shape[1] != 2:
            st.error("❌ The file must have exactly 2 columns.")
            uploaded_file = None  # reset to avoid further processing
        else:
            with st.spinner(''):
                st.write("Preview of uploaded file:")
                st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error reading file: {e}")
    

# Matching Techniques Dropdown with Helper Icon
dropdown, icon = st.columns([0.95, 0.05])  # keep dropdown and helper aligned
with dropdown:
    matching_method = st.selectbox(
        "**Choose a matching technique**:",
        ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
         "Semantic Matching (SentenceTransformer)"]
    )
with icon:
    # Push popover down a bit to alighn with the title
    st.markdown("<br>", unsafe_allow_html=True)
    with st.popover("❓"):
        st.markdown("""
        #### **Choose the matching methodology**
        ##### **➡FuzzyWuzzy**
        It works well for detecting spelling differences, rearrangements, and partial matches, making it effective for cases where text values are similar but not identical.
        - **Ratio** -> When you want a strict comparison of the entire string. Best if the strings are already normalized/cleaned and order matters.
        - **Partial_Ratio** -> When one string may be embedded inside another. Good for matching short forms against long descriptions.
        - **Token_Sort_Ratio** -> When the strings have the same words but in different orders.
        - **Token_Set_Ratio** -> When strings share a common subset of words but one has extra info. Best for messy data with additional descriptive words.
        ##### **➡Semantic Matching**
        - **SentenceTransformer** -> It's effective for understanding context, synonyms, and paraphrases. learned from billions of sentences.
        """)

st.write("You selected:", matching_method)

# Stop Words Input
# Wrap text_area + button in a form
with st.form("stop_words_form"):
    stop_words = st.text_area(
    "**Enter words that should be excluded from the comparison** (comma separated):",
    placeholder="e.g. station, fuel, gas, corp, ltd, inc, group, university, hospital, restaurant"
    )
    st.caption(
    "These are common or irrelevant terms that don’t change the identity of the name:\n"
    "- For gas stations → station, fuel, gas, etc.\n"
    "- For companies → corp, ltd, inc, co, group, etc.\n"
    "- For hospitals → hospital, clinic, medical center, etc.\n"
    )
    submitted = st.form_submit_button(label="Proceed..")

# Process and show the provided stop words after submission
if submitted:
    stop_words_list = [w.strip() for w in stop_words.split(",") if w.strip()]
    st.write("Stop words entered:")
    st.write("➡",", ".join(stop_words_list))

# Process the data if file is uploaded and stop words are submitted
if uploaded_file and submitted:
    # Stage 1: Data Cleaning
    with st.spinner('Stage 1/3: Cleaning data in progress...'):
        # Get the column names
        cols = df.columns[:2].tolist()
        
        # Split into two dataframes
        df1 = df[[cols[0]]].copy()
        df2 = df[[cols[1]]].copy()
        
        # Clean each dataframe separately
        cleaned_df1 = clean_dataframe(df1, [cols[0]], stop_words=stop_words_list)
        cleaned_df2 = clean_dataframe(df2, [cols[1]], stop_words=stop_words_list)
        
        # Save both cleaned dataframes
        cleaned_df1.to_excel('Data/Cleaned_Input/cleaned_df1.xlsx', index=False)
        cleaned_df2.to_excel('Data/Cleaned_Input/cleaned_df2.xlsx', index=False)
        
        st.success('Stage 1/3: Data cleaning completed!')
        
        # Display results in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Cleaned Data for {cols[0]}")
            st.dataframe(cleaned_df1.head())
        with col2:
            st.write(f"Cleaned Data for {cols[1]}")
            st.dataframe(cleaned_df2.head())

    # Stage 2: Exact Matching
    with st.spinner('Stage 2/3: Performing exact matching...'):
        try:
            matched_df, unmatched_df = exact_match(cleaned_df1, cleaned_df2)
            st.success('Stage 2/3: Exact matching completed!')
            st.write(f"Exact/Sorted key matching results:")
            st.dataframe(matched_df.head())
            #unmatched_df.to_excel('Data/Output/unmatched_exact.xlsx', index=False)
            matched_df.to_excel('Data/Output/matched_exact.xlsx', index=False)
        except Exception as e:
            st.error(f"⚠️ Stage 2 failed: {e}")
            st.stop()

    # Stage 3: Fuzzy Matching
    with st.spinner(f'Stage 3/3: Performing fuzzy matching using {matching_method}...'):
        try:
            stage3_matches = fuzzy_match_blocking(
                unmatched_df,   # df1 records left unmatched
                cleaned_df2,    # full df2 reference
                method=matching_method,  
                threshold=80    # adjust threshold as needed
            )
            st.success('Stage 3/3: Fuzzy matching completed!')
            st.write(f"Fuzzy matching({matching_method}) results:")
            st.dataframe(stage3_matches.head())
            stage3_matches.to_excel('Data/Output/matched_fuzzy.xlsx', index=False)
        except Exception as e:
            st.error(f"⚠️ Stage 3 failed: {e}")
            st.stop()
    try:
        final = build_final_output(cleaned_df1,matched_df,stage3_matches)
        st.write(f"Final results:")
        st.dataframe(final.head())
        final.to_excel('Data/Output/matched_final.xlsx', index=False)

        output_buffer = io.BytesIO()
        final.to_excel(output_buffer, index=False, engine="openpyxl")
        output_buffer.seek(0)

        st.download_button(
            label="📥 Download Final Matched File",
            data=output_buffer,
            file_name="matched_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
            st.error(f"⚠️ Final output failed: {e}")
            st.stop()