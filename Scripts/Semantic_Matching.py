import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm
import numpy as np

# Constants
#MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast and efficient model with good performance
MODEL_NAME = 'all-mpnet-base-v2'  

SEMANTIC_MATCH_THRESHOLD = 75  # Score threshold for accepting matches (0-100 scale)

def semantic_match_blocking(unmatched_df, df2, threshold=SEMANTIC_MATCH_THRESHOLD):
    """
    Perform semantic matching using SentenceTransformers with reciprocal validation.
    Similar interface to fuzzy_match_blocking but uses neural embeddings instead.
    
    Args:
        unmatched_df (pd.DataFrame): DataFrame with unmatched records from df1
        df2 (pd.DataFrame): Reference DataFrame to match against
        threshold (float): Minimum score (0-100) to consider a match valid
        
    Returns:
        pd.DataFrame: DataFrame containing the matches found with scores and metadata
    """
    # Get column names
    col1_cleaned = [c for c in unmatched_df.columns if c.endswith("_cleaned")][0]
    col2_cleaned = [c for c in df2.columns if c.endswith("_cleaned")][0]
    
    # Ensure strings and fill NaN
    unmatched_df[col1_cleaned] = unmatched_df[col1_cleaned].fillna("").astype(str)
    df2[col2_cleaned] = df2[col2_cleaned].fillna("").astype(str)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading NLP model '{MODEL_NAME}' on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # Prepare texts for encoding
    df1_texts = unmatched_df[col1_cleaned].tolist()
    df2_texts = df2[col2_cleaned].tolist()
    
    # Generate embeddings
    print("Generating embeddings for reference records...")
    df2_embeddings = model.encode(df2_texts, convert_to_tensor=True, show_progress_bar=True)
    
    print("Generating embeddings for unmatched records...")
    df1_embeddings = model.encode(df1_texts, convert_to_tensor=True, show_progress_bar=True)
    
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
