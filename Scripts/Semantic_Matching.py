import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm
import numpy as np
import streamlit as st

# Constants
MODEL_NAME = 'all-mpnet-base-v2'  
SEMANTIC_MATCH_THRESHOLD = 75  # Score threshold for accepting matches (0-100 scale)

@st.cache_resource(show_spinner="Loading NLP model...")
def load_semantic_model():
    """
    Load and cache the SentenceTransformer model.
    Uses Streamlit's cache_resource to ensure the model is loaded only once
    and reused across all runs.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model

@st.cache_data(show_spinner=False)
def generate_text_embeddings(texts, model_name: str):
    """
    Generate and cache embeddings for a list of texts.
    
    Args:
        texts: List of texts to encode
        model_name: The name of the model, to ensure the cache is specific to the model being used.
    
    Returns:
        torch.Tensor: Tensor containing the embeddings
    """
    model = load_semantic_model()  # Get the cached model
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

def semantic_match_blocking(unmatched_df, df2, threshold=SEMANTIC_MATCH_THRESHOLD, progress_callback=None):
    """
    Perform semantic matching using SentenceTransformers with reciprocal validation.
    Similar interface to fuzzy_match_blocking but uses neural embeddings instead.
    
    Args:
        unmatched_df (pd.DataFrame): DataFrame with unmatched records from df1
        df2 (pd.DataFrame): Reference DataFrame to match against
        threshold (float): Minimum score (0-100) to consider a match valid
        progress_callback (callable): Function to call with progress updates (0-100) and status message
        
    Returns:
        pd.DataFrame: DataFrame containing the matches found with scores and metadata
    """
    # Get column names
    col1_cleaned = [c for c in unmatched_df.columns if c.endswith("_cleaned")][0]
    col2_cleaned = [c for c in df2.columns if c.endswith("_cleaned")][0]
    
    # Ensure strings and fill NaN
    unmatched_df[col1_cleaned] = unmatched_df[col1_cleaned].fillna("").astype(str)
    df2[col2_cleaned] = df2[col2_cleaned].fillna("").astype(str)
    
    # Get model instance (cached)
    if progress_callback:
        progress_callback(5, "🔄 Initializing semantic model...")
    
    model = load_semantic_model()
    model_id = id(model)  # Use model's id for cache invalidation
    
    # Prepare texts, tuples are hashable for the cache
    df1_texts = tuple(unmatched_df[col1_cleaned].tolist())
    df2_texts = tuple(df2[col2_cleaned].tolist())
    
    # Generate embeddings with caching, passing the MODEL_NAME constant
    if progress_callback:
        progress_callback(15, "🔄 Generating embeddings for reference data...")
    
    # Pass MODEL_NAME as the cache key dependency
    df2_embeddings = generate_text_embeddings(df2_texts, model_name=MODEL_NAME)
    
    if progress_callback:
        progress_callback(40, "🔄 Generating embeddings for unmatched records...")
    
    df1_embeddings = generate_text_embeddings(df1_texts, model_name=MODEL_NAME)
    
    if progress_callback:
        progress_callback(65, "🔄 Performing semantic search...")
    
    # Forward search (df1 -> df2)
    print("\nPerforming forward search...")
    forward_results = util.semantic_search(df1_embeddings, df2_embeddings, top_k=1)
    
    # Reverse search (df2 -> df1) for validation
    print("Performing reverse search for validation...")
    reverse_results = util.semantic_search(df2_embeddings, df1_embeddings, top_k=1)
    
    # Build reverse lookup map for efficient validation
    reverse_lookup = {
        idx: results[0]['corpus_id'] 
        for idx, results in enumerate(reverse_results)
    }
    
    # Process reciprocal matches
    matches = []
    for df1_idx, results in enumerate(forward_results):
        if not results:
            continue
        
        # Get match details
        df2_idx = results[0]['corpus_id']
        score = round(float(results[0]['score']) * 100, 2)  # Convert to 0-100 scale
        
        # Check reciprocal match
        best_df1_for_df2 = reverse_lookup.get(df2_idx)
        
        # Only keep reciprocal matches above threshold
        if best_df1_for_df2 == df1_idx and score >= threshold:
            matches.append({
                'df1_index': df1_idx,
                'df2_index': df2_idx,
                'match_score': score
            })
    
    if not matches:
        return pd.DataFrame()
    
    # Create matches DataFrame
    df_matches = pd.DataFrame(matches)
    
    # Merge with original DataFrames to get full records
    result = pd.merge(
        unmatched_df.reset_index(),
        df_matches,
        left_index=True,
        right_on='df1_index'
    )
    
    result = pd.merge(
        result,
        df2.add_prefix(''),
        left_on='df2_index',
        right_index=True,
        how='left'
    )
    
    # Add match type
    result['match_type'] = 'semantic'
    
    # Clean up temporary columns
    result = result.drop(columns=['df1_index', 'df2_index'])
    
    return result
