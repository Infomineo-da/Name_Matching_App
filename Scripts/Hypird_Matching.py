
import pandas as pd
from Fuzzy_Matching import fuzzy_match_blocking
from Semantic_Matching import semantic_match_blocking

def hybrid_match_blocking(unmatched_df, df2, threshold=80, 
                          fuzzy_method="token_set_ratio",
                          semantic_threshold=75,
                          progress_callback=None):
    """
    1. Run fuzzy matching.
    2. Correct its output column names.
    3. Identify records that are STILL unmatched.
    4. Run semantic matching ONLY on the remaining records.
    5. Correct its output column names.
    6. Combine the clean results.
    """
    if unmatched_df.empty:
        return pd.DataFrame()

    # --- Stage 1: Fuzzy Matching ---
    if progress_callback:
        progress_callback(5, "🔄 Stage 1/2: Running fuzzy matching...")
    
    fuzzy_matches = fuzzy_match_blocking(
        unmatched_df, df2,
        method=fuzzy_method,
        threshold=threshold,
        progress_callback=None
    )

    # The underlying functions rename 'unique_id' to 'unique_id_x' after their merge.
    # We rename it back here to make the output consistent.
    if not fuzzy_matches.empty and 'unique_id_x' in fuzzy_matches.columns:
        fuzzy_matches.rename(columns={'unique_id_x': 'unique_id'}, inplace=True)
    
    # --- Stage 2: Semantic Matching on the Remainder ---
    
    # Now that 'unique_id' is present, we can safely identify the records that were matched.
    if not fuzzy_matches.empty:
        matched_in_stage1_ids = fuzzy_matches['unique_id'].unique()
    else:
        matched_in_stage1_ids = []

    # Create a new DataFrame of records that are STILL unmatched
    still_unmatched_df = unmatched_df[~unmatched_df['unique_id'].isin(matched_in_stage1_ids)].copy()

    semantic_matches = pd.DataFrame() # Initialize an empty DataFrame for the case where it's not needed
    if not still_unmatched_df.empty:
        if progress_callback:
            progress_callback(50, f"🔄 Stage 2/2: Running semantic matching on {len(still_unmatched_df):,} remaining records...")
        
        semantic_matches = semantic_match_blocking(
            still_unmatched_df, df2,
            threshold=semantic_threshold,
            progress_callback=None
        )

        # We perform the same column name correction on the semantic results.
        if not semantic_matches.empty and 'unique_id_x' in semantic_matches.columns:
            semantic_matches.rename(columns={'unique_id_x': 'unique_id'}, inplace=True)

    # --- Combine Results ---
    if progress_callback:
        progress_callback(95, "✅ Combining results...")

    # Both DataFrames now have a consistent 'unique_id' column, so they can be safely combined.
    final_hybrid_matches = pd.concat([fuzzy_matches, semantic_matches], ignore_index=True)

    return final_hybrid_matches.reset_index(drop=True)