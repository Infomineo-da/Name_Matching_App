import streamlit as st
import pandas as pd
import io
from Data_Cleaning import clean_dataframe
from Fuzzy_Matching import exact_match, fuzzy_match_blocking, build_final_output
from Semantic_Matching import semantic_match_blocking



st.set_page_config(page_title="Text Matching App 🔍", layout="wide")

# Title
st.title("Text Matching App 🔍")

# Upload Section
uploaded_file = st.file_uploader("**Upload your Excel file**: containing two **text** columns to match.", type=["xlsx", "xls"])

# Preview uploaded file
if uploaded_file:
    try:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)

        # HANDLING Empty File
        if df.empty:
            st.error("❌ The file is empty.")
            uploaded_file = None
            st.stop()
            
        # Function to check if a column is text type
        def is_text_column(series):
            return series.dtype == 'object' or series.dtype == 'string'
            
        # Get list of text columns
        text_columns = [col for col in df.columns if is_text_column(df[col])]
        
        if len(text_columns) < 2:
            st.error("❌ The file must have at least 2 text columns. Please check your data types.")
            st.write("Found text columns:", ", ".join(text_columns) if text_columns else "None")
            non_text = [f"{col} ({df[col].dtype})" for col in df.columns if col not in text_columns]
            st.write("Non-text columns:", ", ".join(non_text))
            uploaded_file = None
            st.stop()

        # Let user select which 2 columns to use
        if len(text_columns) == 2:
            df = df[text_columns]
        else:  # More than 2 text columns
            st.info("ℹ️ Please select exactly two text columns to match.")
            selected_cols = st.multiselect(
                "**Select exactly two text columns for matching:**",
                text_columns,  # Only show text columns as options
                default=text_columns[:] if len(text_columns) >= 2 else text_columns
            )
            if len(selected_cols) != 2:
                st.warning("Please select exactly two columns to proceed.")
                uploaded_file = None
                st.stop()
            else:
                # Verify selected columns are text type (double check)
                if not all(is_text_column(df[col]) for col in selected_cols):
                    st.error("❌ All selected columns must be text type.")
                    uploaded_file = None
                    st.stop()
                df = df[selected_cols]

        with st.spinner(''):
            st.write("Preview of uploaded file:")
            st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    

# Matching Techniques Dropdown with Helper Icon
dropdown, score, icon = st.columns([0.80, 0.15, 0.05])  # keep dropdown and helper aligned
with dropdown:
    matching_method = st.selectbox(
        "**Choose a matching technique**:click on the ❗ icon for details.",
        ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
         "Semantic Matching"]
    )
with score:
    score=st.number_input("Minimum score threshold.", min_value=60, max_value=100, value=75, step=1)
with icon:
    # Push popover down a bit to alighn with the title
    st.markdown("<br>", unsafe_allow_html=True)
    with st.popover("❗",help="Description of matching methods"):
        st.markdown("""
        #### **Choose the matching methodology**
        ##### **➡ FuzzyWuzzy**
        It works well for detecting spelling differences, rearrangements, and partial matches, making it effective for cases where text values are similar but not identical.
        - **Ratio** -> When you want a strict comparison of the entire string. Best if the strings are already normalized/cleaned and order matters.
        - **Partial_Ratio** -> When one string may be embedded inside another. Good for matching short forms against long descriptions.
        - **Token_Sort_Ratio** -> When the strings have the same words but in different orders.
        - **Token_Set_Ratio** -> When strings share a common subset of words but one has extra info. Best for messy data with additional descriptive words.
        ##### **➡ Semantic Matching**
        - **SentenceTransformer** -> It's effective for understanding context, synonyms, and paraphrases. learned from billions of sentences.
        ##### **Disclaimer**: 
        Sentence transformers capture semantic meaning but may over-match by treating related concepts as equivalent, leading to false positives. Fuzzy matching, on the other hand, focuses on text similarity but may under-match when the same concept is expressed in different wording.
        """)

st.write(f"You selected: **{matching_method}** with threshold of **{score}**")

# Stop Words Input
# Wrap text_area + button in a form
with st.form("stop_words_form"):
    stop_words = st.text_area(
    "**Ignore words when comparing names**: Add words you don’t want to affect the match (comma separated).",
    placeholder="e.g. station, fuel, gas, corp, ltd, inc, group, university, hospital, restaurant"
    )
    st.caption(
    "These are common or generic words that don’t change the actual name:\n"
    "- For gas stations → station, fuel, gas, etc.\n"
    "- For companies → corp, ltd, inc, co, group, etc.\n"
    "- For hospitals → hospital, clinic, medical center, etc.\n"
    "- You may also ignore common words like -> the, in, a, of, over, etc. but be careful! Sometimes they are part of the real name"
    )
    submitted = st.form_submit_button(label="Proceed..")

# Process and show the provided stop words after submission
if submitted:
    stop_words_list = [w.strip() for w in stop_words.split(",") if w.strip()]
    st.write("Ignored words:")
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
        
        # Display cleaning statistics
        st.write("📊 Cleaning Statistics:")
        col1, col2 = st.columns(2)
        with col1:
            # Count only non-empty original records
            total_records = df1[cols[0]].notna().sum()
            cleaned_records = len(cleaned_df1)
            removed_records = total_records - cleaned_records
            st.metric(f"Column 1: {cols[0]}", 
                     f"{cleaned_records:,} records cleaned",
                     f"{removed_records:,} removed")
        with col2:
            # Count only non-empty original records
            total_records = df2[cols[1]].notna().sum()
            cleaned_records = len(cleaned_df2)
            removed_records = total_records - cleaned_records
            st.metric(f"Column 2: {cols[1]}", 
                     f"{cleaned_records:,} records cleaned",
                     f"{removed_records:,} removed")

    # Stage 2: Exact Matching
    with st.spinner('Stage 2/3: Performing exact matching...'):
        try:
            matched_df, unmatched_df = exact_match(cleaned_df1, cleaned_df2)
            st.success('Stage 2/3: Exact matching completed!')
            
            # Display exact matching statistics
            st.write("📊 Exact Matching Statistics:")
            col1, col2 = st.columns(2)
            with col1:
                primary_matches = len(matched_df[matched_df['match_type'] == 'primary key'])
                st.metric("Exact Matches", f"{primary_matches:,} records")
            with col2:
                sorted_matches = len(matched_df[matched_df['match_type'] == 'sorted key'])
                st.metric("Sorted Key Matches", f"{sorted_matches:,} records")
            
            matched_df.to_excel('Data/Output/matched_exact.xlsx', index=False)
        except Exception as e:
            st.error(f"⚠️ Stage 2 failed: {e}")
            st.stop()
    # Stage 3: Advanced Matching (Fuzzy or Semantic)
    with st.spinner(f'Stage 3/3: Performing {matching_method} matching...'):
        try:
            # Set up progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if matching_method == "Semantic Matching":
                status_text.text("🔄 Loading model and generating embeddings...")
                stage3_matches = semantic_match_blocking(
                    unmatched_df,   # df1 records left unmatched
                    cleaned_df2,    # full df2 reference
                    threshold=score,    # adjust threshold as needed
                    progress_callback=lambda p, msg: (progress_bar.progress(p), status_text.text(msg))
                )
                match_type = "semantic"
            else:
                status_text.text("🔄 Preparing fuzzy matching...")
                stage3_matches = fuzzy_match_blocking(
                    unmatched_df,   # df1 records left unmatched
                    cleaned_df2,    # full df2 reference
                    method=matching_method,  
                    threshold=score,    # adjust threshold as needed
                    progress_callback=lambda p, msg: (progress_bar.progress(p), status_text.text(msg))
                )
                match_type = "fuzzy"
            
            # Clear progress indicators after completion
            progress_bar.empty()
            status_text.empty()
            
            st.success('Stage 3/3: Advanced matching completed!')
            
            # Display advanced matching statistics
            st.write(f"📊 {match_type.capitalize()} Matching Statistics:")
            col1, col2, col3 = st.columns(3)
            with col1:
                advanced_matches = len(stage3_matches)
                st.metric(f"{match_type.capitalize()} Matches", 
                         f"{advanced_matches:,} records")
            with col2:
                if not stage3_matches.empty:
                    avg_score = stage3_matches['match_score'].mean()
                    st.metric("Average Match Score", 
                            f"{avg_score:.1f}%")
            with col3:
                if not stage3_matches.empty:
                    high_quality = len(stage3_matches[stage3_matches['match_score'] >= 90])
                    st.metric("High Quality Matches (≥90%)", 
                            f"{high_quality:,} records")
            
            stage3_matches.to_excel(f'Data/Output/matched_{match_type}.xlsx', index=False)
        except Exception as e:
            st.error(f"⚠️ Stage 3 failed: {e}")
            st.stop()
    try:
        final = build_final_output(cleaned_df1,matched_df,stage3_matches)
        
        # Calculate and display total matching statistics
        st.write("📊 Overall Matching Results:")
        total_records = len(cleaned_df1)
        matched_records = len(final[final['match_type'] != 'unmatched'])
        match_rate = (matched_records / total_records * 100) if total_records > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{total_records:,}")
        with col2:
            st.metric("Total Matched", f"{matched_records:,}")
        with col3:
            st.metric("Match Rate", f"{match_rate:.1f}%")
            
        st.write("---")  # Add a visual separator
        st.write("Final results preview:")
        st.dataframe(final.head())
        final.to_excel('Data/Output/matched_final.xlsx', index=False)

        output_buffer = io.BytesIO()
        final.to_excel(output_buffer, index=False, engine="openpyxl")
        output_buffer.seek(0)

        st.download_button(
            label="📥 Download Final Matched File",
            data=output_buffer,
            file_name=f'matched_{matching_method}.xlsx',
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
            st.error(f"⚠️ Final output failed: {e}")
            st.stop()