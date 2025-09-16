from fuzzywuzzy import fuzz
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd


def exact_match(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:

    col1_cleaned = [c for c in df1.columns if c.endswith("_cleaned")][0]
    col2_cleaned = [c for c in df2.columns if c.endswith("_cleaned")][0]

    col1_sorted = [c for c in df1.columns if c.endswith("_sorted")][0]
    col2_sorted = [c for c in df2.columns if c.endswith("_sorted")][0]

    # Filter out empty or NaN values before matching
    df1_valid = df1[df1[col1_cleaned].notna() & (df1[col1_cleaned] != "")].copy()
    df2_valid = df2[df2[col2_cleaned].notna() & (df2[col2_cleaned] != "")].copy()
    
    # Add unique identifiers to track matches
    df1_valid['temp_id'] = range(len(df1_valid))
    df2_valid['temp_id'] = range(len(df2_valid))

    # --- Stage 1: Match on cleaned column ---
    stage1_matches = pd.merge(
        df1_valid,
        df2_valid,
        left_on=col1_cleaned,
        right_on=col2_cleaned,
        how="inner"
    )
    
    # Remove duplicates keeping the best match for both df1 and df2
    stage1_matches = stage1_matches.sort_values('temp_id_x')
    #stage1_matches = stage1_matches.drop_duplicates(subset=['temp_id_x'], keep='first')
    #stage1_matches = stage1_matches.drop_duplicates(subset=['temp_id_y'], keep='first')
    stage1_matches["match_score"] = 100
    stage1_matches["match_type"] = "primary key"

    # Track matched records from df2
    matched_df2_ids = stage1_matches['temp_id_y'].unique()
    
    # Identify unmatched records from df1 after Stage 1
    unmatched_stage1 = df1_valid[~df1_valid['temp_id'].isin(stage1_matches['temp_id_x'])]
    df2_remaining = df2_valid[~df2_valid['temp_id'].isin(matched_df2_ids)]

    # --- Stage 2: Match remaining unmatched on sorted column ---
    stage2_matches = pd.merge(
        unmatched_stage1[unmatched_stage1[col1_sorted].notna() & (unmatched_stage1[col1_sorted] != "")],
        df2_remaining,
        left_on=col1_sorted,
        right_on=col2_sorted,
        how="inner"
    )
    
    # Remove duplicates keeping the best match for both sides
    stage2_matches = stage2_matches.sort_values('temp_id_x')
    #stage2_matches = stage2_matches.drop_duplicates(subset=['temp_id_x'], keep='first')
    #stage2_matches = stage2_matches.drop_duplicates(subset=['temp_id_y'], keep='first')
    stage2_matches["match_score"] = 99
    stage2_matches["match_type"] = "sorted key"

    # --- Final unmatched after Stage 2 ---
    unmatched_final = df1[~df1_valid['temp_id'].isin(
        pd.concat([stage1_matches['temp_id_x'], stage2_matches['temp_id_x']], ignore_index=True)
    )]

    # --- Combine matches ---
    # Drop temporary columns before combining
    stage1_matches = stage1_matches.drop(['temp_id_x', 'temp_id_y'], axis=1)
    stage2_matches = stage2_matches.drop(['temp_id_x', 'temp_id_y'], axis=1)
    all_exact_matches = pd.concat([stage1_matches, stage2_matches], ignore_index=True)

    # Final cleanup of any remaining empty matches
    all_exact_matches = all_exact_matches[
        all_exact_matches[col1_cleaned].notna() & 
        (all_exact_matches[col1_cleaned] != "") &
        all_exact_matches[col2_cleaned].notna() & 
        (all_exact_matches[col2_cleaned] != "")
    ]

    # Clean unmatched records
    unmatched_final = unmatched_final[
        unmatched_final[col1_cleaned].notna() & 
        (unmatched_final[col1_cleaned] != "")
    ]
    unmatched_final = unmatched_final.drop_duplicates(subset=[col1_cleaned], keep='first')

    return all_exact_matches, unmatched_final


##############################################################################


def build_blocks(df, col_cleaned, prefix_len=4):
    """Create blocking dictionary for faster lookups."""
    blocks = defaultdict(list)
    for index, name in df[col_cleaned].fillna("").astype(str).items():
        if name.strip():
            block_key = name[:prefix_len]
            blocks[block_key].append(index)
    return blocks


def find_best_fuzzy_match(client_row_tuple, df2, blocks, col1_cleaned, col2_cleaned, method, threshold):
    """
    Finds the best fuzzy match for a single row in df1 against df2 using blocking.
    Penalizes subset matches that hit score 100.
    """
    client_index, client_cleaned, client_sorted, client_id = client_row_tuple

    if not client_cleaned or not isinstance(client_cleaned, str):
        return None

    block_key = client_cleaned[:4]
    candidate_indices = blocks.get(block_key)
    if not candidate_indices:
        return None

    candidate_names = df2.loc[candidate_indices, col2_cleaned]

    # choose scoring function
    scorers = {
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "token_set_ratio": fuzz.token_set_ratio
    }
    scorer = scorers.get(method, fuzz.token_set_ratio)

    best_score = 0
    best_match_index = -1

    for idx, cand in candidate_names.items():
        score = scorer(client_cleaned, cand)

        # Penalize subset 100 matches
        if score == 100 and client_cleaned != cand:
            if len(client_cleaned.split()) != len(cand.split()):
                score = 80

        if score > best_score:
            best_score = score
            best_match_index = idx

    if best_score >= threshold:
        return (client_id, best_match_index, best_score)
    else:
        return None


def fuzzy_match_blocking(unmatched_df, df2, method="token_set_ratio", threshold=80):
    """
    Fuzzy match unmatched_df (df1) against df2 with blocking and parallelization.
    """
    col1_cleaned = [c for c in unmatched_df.columns if c.endswith("_cleaned")][0]
    col1_sorted = [c for c in unmatched_df.columns if c.endswith("_sorted")][0]
    col2_cleaned = [c for c in df2.columns if c.endswith("_cleaned")][0]

    # Ensure strings
    unmatched_df[col1_cleaned] = unmatched_df[col1_cleaned].fillna("").astype(str)
    df2[col2_cleaned] = df2[col2_cleaned].fillna("").astype(str)

    # Build blocks on df2
    blocks = build_blocks(df2, col2_cleaned)

    # Run parallel fuzzy matching
    fuzzy_results = Parallel(n_jobs=-1)(
        delayed(find_best_fuzzy_match)(
            row, df2, blocks, col1_cleaned, col2_cleaned, method, threshold
        ) for row in tqdm(
            unmatched_df[[col1_cleaned, col1_sorted, "unique_id"]].itertuples(),
            total=len(unmatched_df),
            desc="Fuzzy Matching"
        )
    )

    # Filter results
    successful = [res for res in fuzzy_results if res is not None]

    if not successful:
        return pd.DataFrame()

    df_fuzzy = pd.DataFrame(successful, columns=["unique_id", "df2_index", "match_score"])

    # Merge back to get full match info
    stage3_matches_temp = pd.merge(unmatched_df, df_fuzzy, on="unique_id")
    stage3_matches = pd.merge(
        stage3_matches_temp,
        df2.add_prefix(""),
        left_on="df2_index",
        right_index=True,
        how="left"
    )
    stage3_matches["match_type"] = f"fuzzy_{method}"
    return stage3_matches.drop(columns=["df2_index"])

def build_final_output(df1, matched_exact, matched_fuzzy):
    """
    Combine exact and fuzzy matches with unmatched records into a clean final output.
    Only includes original columns, cleaned columns, match score and type.
    """
    # Get column names from df1
    col1_original = [c for c in df1.columns if not c.endswith(('_cleaned', '_sorted', 'unique_id'))][0]
    col1_cleaned = [c for c in df1.columns if c.endswith('_cleaned')][0]
    
    # Initialize empty DataFrame with desired columns
    final_output = pd.DataFrame(columns=[
        col1_original, col1_cleaned,
        'matched_original', 'matched_cleaned',
        'match_score', 'match_type'
    ])
    
    # Process exact matches
    if not matched_exact.empty:
        col2_original = [c for c in matched_exact.columns if not c.endswith(('_cleaned', '_sorted')) and not c.startswith('unique_id')][1]
        col2_cleaned = [c for c in matched_exact.columns if c.endswith('_cleaned')][1]
        
        exact_df = pd.DataFrame({
            col1_original: matched_exact[col1_original],
            col1_cleaned: matched_exact[col1_cleaned],
            'matched_original': matched_exact[col2_original],
            'matched_cleaned': matched_exact[col2_cleaned],
            'match_score': matched_exact['match_score'],
            'match_type': matched_exact['match_type']
        })
        final_output = pd.concat([final_output, exact_df], ignore_index=True)
    
    # Process fuzzy/semantic matches
    if not matched_fuzzy.empty:
        # First find the cleaned column that isn't col1_cleaned
        all_cleaned_cols = [c for c in matched_fuzzy.columns if c.endswith('_cleaned')]
        col2_cleaned = next(col for col in all_cleaned_cols if col != col1_cleaned)
        
        # Find the original column by removing the '_cleaned' suffix
        col2_base = col2_cleaned.replace('_cleaned', '')
        col2_original = next(col for col in matched_fuzzy.columns 
                           if not col.endswith(('_cleaned', '_sorted')) 
                           and col.replace('_cleaned', '') == col2_base)
        
        fuzzy_df = pd.DataFrame({
            col1_original: matched_fuzzy[col1_original],
            col1_cleaned: matched_fuzzy[col1_cleaned],
            'matched_original': matched_fuzzy[col2_original],
            'matched_cleaned': matched_fuzzy[col2_cleaned],
            'match_score': matched_fuzzy['match_score'],
            'match_type': matched_fuzzy['match_type']
        })
        final_output = pd.concat([final_output, fuzzy_df], ignore_index=True)
    
    # Get all matched records to identify unmatched ones
    matched_records = set(final_output[col1_original])
    
    # Process unmatched records
    unmatched = df1[~df1[col1_original].isin(matched_records)]
    if not unmatched.empty:
        unmatched_df = pd.DataFrame({
            col1_original: unmatched[col1_original],
            col1_cleaned: unmatched[col1_cleaned],
            'matched_original': None,
            'matched_cleaned': None,
            'match_score': 0,
            'match_type': 'unmatched'
        })
        final_output = pd.concat([final_output, unmatched_df], ignore_index=True)
    final_output.drop_duplicates(inplace=True)
    return final_output


"""
# Test exact match logic
df1 = pd.read_excel('Data/Cleaned_Input/Cleaned_df1.xlsx')
df2 = pd.read_excel('Data/Cleaned_Input/Cleaned_df2.xlsx')
matched, unmatched = exact_match(df1, df2)

matched.to_excel('Data/Output/matched_exact.xlsx', index=False)
unmatched.to_excel('Data/Output/unmatched_exact.xlsx', index=False)
"""


"""
# Test fuzzy match logic
df1 = pd.read_excel('Data/Output/unmatched_exact.xlsx')
df2 = pd.read_excel('Data/Cleaned_Input/Cleaned_df2.xlsx')

# Ensure unique ID exists
if "client_unique_id" not in df1.columns:
    df1 = df1.reset_index(drop=True).reset_index().rename(columns={"index": "client_unique_id"})

stage3_matches = fuzzy_match_blocking(
    df1,   # df1 records left unmatched
    df2,               # full df2 reference
    method="token_set_ratio",  # you can change to "ratio", "partial_ratio", etc.
    threshold=80       # adjust threshold as needed
)
stage3_matches.to_excel('Data/Output/matched_fuzzy.xlsx', index=False)
"""

"""
# Test final output
df1 = pd.read_excel('Data/Cleaned_Input/cleaned_df1.xlsx')
exact = pd.read_excel('Data/Output/matched_exact.xlsx')
fuzzy = pd.read_excel('Data/Output/matched_fuzzy.xlsx')


final = build_final_output(
    df1,
    exact,
    fuzzy
)
final.to_excel('Data/Output/matched_final.xlsx', index=False)
"""

