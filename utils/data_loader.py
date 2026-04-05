import torch
from torch.utils.data import Dataset
import torch.nn as nn
from datetime import datetime
import numpy as np


def pad_collate_reddit(batch):
    s_y = [item[0] for item in batch]
    cur_su_y = [item[1] for item in batch]
    b_y = [item[2] for item in batch]
    res_y = [item[3] for item in batch]
    tweets = [torch.nan_to_num(item[4]) for item in batch]
    time_interval = [item[5] for item in batch]
    raw_timestamps = [item[6] for item in batch]
    user_id = [item[7] for item in batch]

    post_num = [len(x) for x in b_y]
    cur_su_y = nn.utils.rnn.pad_sequence(cur_su_y, batch_first=True, padding_value=0)
    b_y = nn.utils.rnn.pad_sequence(b_y, batch_first=True, padding_value=0)
    res_y = nn.utils.rnn.pad_sequence(res_y, batch_first=True, padding_value=0)
    tweets = nn.utils.rnn.pad_sequence(tweets, batch_first=True, padding_value=0)
    time_interval = nn.utils.rnn.pad_sequence(time_interval, batch_first=True, padding_value=0)

    post_num = torch.tensor(post_num)
    s_y = torch.tensor(s_y)
    user_id = torch.tensor(user_id)
    return [s_y, cur_su_y, b_y, res_y, post_num, tweets, time_interval, raw_timestamps, user_id]


def get_adjacent_time_intervals(timestamps):
    """Compute time intervals (in days) between consecutive posts."""
    if isinstance(timestamps[0], str):
        timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]

    intervals = [0.0]
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        intervals.append(delta.total_seconds() / (24 * 3600))

    return intervals


def get_timestamp(x):
    def change_utc(x):
        try:
            x = str(datetime.fromtimestamp(int(x) / 1000))
            return x
        except Exception:
            return str(x)

    timestamp = [datetime.timestamp(datetime.strptime(change_utc(t), "%Y-%m-%d %H:%M:%S")) for t in x]
    time_interval = (timestamp[-1] - np.array(timestamp))
    return time_interval


class RedditDataset(Dataset):
    def __init__(self, s_y, cur_su_y, b_y, res_y, tweets, timestamp, user_id, days=30):
        super().__init__()
        self.s_y = s_y
        self.cur_su_y = cur_su_y
        self.b_y = b_y
        self.res_y = res_y
        self.tweets = tweets
        self.timestamp = timestamp
        self.user_id = user_id
        self.days = days

    def __len__(self):
        return len(self.s_y)

    def __getitem__(self, item):
        s_y = torch.tensor(self.s_y[item], dtype=torch.long)
        user_id = torch.tensor(self.user_id[item], dtype=torch.long)

        if self.days > len(self.tweets[item]):
            cur_su_y = torch.tensor(self.cur_su_y[item], dtype=torch.long)
            b_y = torch.tensor(self.b_y[item], dtype=torch.long)
            res_y = torch.tensor(self.res_y[item], dtype=torch.long)
            tweets = torch.tensor(self.tweets[item], dtype=torch.float32)
            raw_timestamps = self.timestamp[item]
            time_interval = get_adjacent_time_intervals(raw_timestamps)
            time_interval = torch.tensor(time_interval, dtype=torch.float32)
        else:
            cur_su_y = torch.tensor(self.cur_su_y[item][:self.days], dtype=torch.long)
            b_y = torch.tensor(self.b_y[item][:self.days], dtype=torch.long)
            res_y = torch.tensor(self.res_y[item][:self.days], dtype=torch.long)
            tweets = torch.tensor(self.tweets[item][:self.days], dtype=torch.float32)
            raw_timestamps = self.timestamp[item][:self.days]
            time_interval = get_adjacent_time_intervals(raw_timestamps)
            time_interval = torch.tensor(time_interval, dtype=torch.float32)

        return [s_y, cur_su_y, b_y, res_y, tweets, time_interval, raw_timestamps, user_id]
