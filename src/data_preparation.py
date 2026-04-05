"""
Data preparation script: builds sliding-window timelines from the raw dataset
and generates sentence embeddings for each post sequence.

Usage:
    python src/data_preparation.py --input <path_to_excel> --window <window_size> --output <output_pkl>
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


RISK_MAPPING = {
    'indicator': 0,
    'ideation': 1,
    'behavior': 2,
    'attempt': 3
}

RISK_COLUMNS = [
    "hopelessness",
    "prior self-harm or suicidal thought/attempt",
    "poor social support",
    "suicide means (with access)"
]

RESILIENCE_COLUMNS = [
    "coping strategy",
    "psychological capital",
    "sense of responsibility",
    "meaning in life"
]


def create_timeline(group, window_size):
    """Build sliding-window timeline records for a single user group."""
    if len(group) < window_size + 1:
        return pd.DataFrame()

    timelines = []
    for i in range(len(group) - window_size):
        window = group.iloc[i:i + window_size + 1]
        posts = window['post'].iloc[:window_size].tolist()
        risk_factors = [
            np.clip(np.array(window.iloc[j][RISK_COLUMNS]), None, 1)
            for j in range(window_size)
        ]
        resilience_factors = [
            np.clip(np.array(window.iloc[j][RESILIENCE_COLUMNS]), None, 1)
            for j in range(window_size)
        ]
        timestamps = window['created_utc'].iloc[:window_size].tolist()
        historic_risk = [
            RISK_MAPPING[risk]
            for risk in window['suicide risk'].iloc[:window_size].tolist()
        ]

        timelines.append({
            'user_id': group.name,
            'author': group.name,
            'sb_post': posts,
            'cur_bp_y': risk_factors,
            'cur_bp_res': resilience_factors,
            'created_utc': timestamps,
            'cur_su_y': historic_risk,
            'fu_30_su_y': window['suicide risk'].iloc[window_size]
        })

    return pd.DataFrame(timelines) if timelines else pd.DataFrame()


def generate_embeddings(texts, model):
    """Generate sentence embeddings for each post timeline."""
    post_embedding_list = []
    for timeline in tqdm(texts, desc="Generating embeddings", unit="timeline"):
        post_embeddings = [model.encode(post) for post in timeline]
        post_embedding_list.append(post_embeddings)
    return post_embedding_list


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset with sliding-window timelines and sentence embeddings.")
    parser.add_argument("--input", type=str, required=True, help="Path to input Excel file")
    parser.add_argument("--window", type=int, default=1, help="Sliding window size")
    parser.add_argument("--output", type=str, required=True, help="Path to output .pkl file")
    parser.add_argument("--model", type=str, default="sentence-transformers/nli-roberta-large",
                        help="Sentence transformer model name")
    args = parser.parse_args()

    df = pd.read_excel(args.input, index_col=0)

    df['created_utc'] = (
        pd.to_datetime('2017-01-01') +
        pd.to_timedelta(df["days_difference"], unit='D')
    ).dt.floor('s')

    df = df.sort_values(['users', 'created_utc'])

    result = df.groupby('users').apply(
        lambda g: create_timeline(g, args.window)
    ).reset_index(drop=True)

    if result.empty:
        result = pd.DataFrame(columns=[
            'author', 'user_id', 'created_utc', 'cur_su_y',
            'sb_post', 'cur_bp_y', 'cur_bp_res', 'fu_30_su_y'
        ])
    else:
        user_id_mapping = {user: i for i, user in enumerate(result['user_id'].unique())}
        result['user_id'] = result['user_id'].map(user_id_mapping)
        result['fu_30_su_y'] = result['fu_30_su_y'].map(RISK_MAPPING)
        columns_order = ['author', 'user_id', 'created_utc', 'cur_su_y', 'sb_post', 'cur_bp_y', 'cur_bp_res', 'fu_30_su_y']
        result = result[columns_order]

    sv_model = SentenceTransformer(args.model)
    post_embedding = generate_embeddings(result['sb_post'].tolist(), sv_model)
    result['sb_1024'] = post_embedding

    result.to_pickle(args.output)
    print(f"Saved {len(result)} records to {args.output}")


if __name__ == "__main__":
    main()
