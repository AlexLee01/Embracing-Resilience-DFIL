import numpy as np
import pandas as pd
from datetime import datetime
from pprint import pprint
from pathlib import Path
import pickle
from sklearn.metrics import classification_report


def gr_metrics(op, t):
    TP = (op == t).sum()
    FN = (t > op).sum()
    FP = (t < op).sum()

    GP = TP / (TP + FP)
    GR = TP / (TP + FN)
    FS = 2 * GP * GR / (GP + GR)
    OE = (t - op > 1).sum() / op.shape[0]

    return GP, GR, FS, OE


def evaluation(config, outputs, _type, y_true_col, y_pred_col, user_id_col):
    if _type == 'fs':
        if config['s_y_num'] == 4:
            label_names = ['su_indicator', 'su_ideation', 'su_behavior', 'su_attempt']
        elif config['s_y_num'] == 3:
            label_names = ['su_indicator', 'su_ideation', 'su_behav + att']
        elif config['s_y_num'] == 2:
            label_names = ['su_indicator', 'su_id+beh+att']

    if _type == 'bd':
        label_names = [
            "hopelessness",
            "prior self-harm or suicidal thought/attempt",
            "poor social support",
            "suicide means (with access)"
        ]
    if _type == 'res':
        label_names = [
            'coping_strategy',
            'psychological_capital',
            'sense_of_responsibility',
            'meaning_in_life'
        ]

    y_true = []
    y_pred = []
    user_id = []

    for i in outputs:
        y_true += i[y_true_col]
        y_pred += i[y_pred_col]
        user_id += i[user_id_col]

    y_true = np.asanyarray(y_true)
    y_pred = np.asanyarray(y_pred)
    user_id = np.asanyarray(user_id)

    pred_dict = {
        'user_id': user_id,
        'y_true': y_true,
        'y_pred': y_pred
    }

    if _type == 'fs':
        print("GP, GR, FS, OE:", gr_metrics(y_pred, y_true))

    print("-------test_report-------")
    metrics_dict = classification_report(
        y_true, y_pred,
        zero_division=1,
        target_names=label_names,
        output_dict=True
    )
    df_result = pd.DataFrame(metrics_dict).transpose()
    pprint(df_result)

    print("-------save test_report-------")
    save_time = datetime.now().__format__("%m%d_%H%M%S%Z")
    save_path = "../result/"
    Path(f"{save_path}/pred").mkdir(parents=True, exist_ok=True)

    df_result.to_csv(f'../result/{save_time}_{_type}.csv')
    with open(f'{save_path}pred/{save_time}_{_type}_pred.pkl', "wb") as outfile:
        pickle.dump(pred_dict, outfile)
