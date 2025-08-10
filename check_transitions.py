import pandas as pd

df = pd.read_csv('rawdata/by_comp_1second/AAPL_201901_202507_1second.csv.gz', compression='gzip', parse_dates=['datetime'])
df['time_diff'] = df['datetime'].diff()

# Check the transition between days
transition_idx = df[df['datetime'].dt.date != df['datetime'].shift(1).dt.date].index
print('Transition between days occurs at index:', transition_idx.tolist())

# Show records around the transition
for idx in transition_idx:
    if idx > 0:
        print(f'\nRecords around index {idx}:')
        print(df.iloc[idx-2:idx+2][['datetime', 'close', 'volume']])
        print(f'Time gap: {df.iloc[idx]["time_diff"].total_seconds()/3600:.1f} hours')

# Check first and last records of each day
for day in df['datetime'].dt.date.unique():
    day_df = df[df['datetime'].dt.date == day]
    print(f'\nDay {day}:')
    print(f'  First: {day_df.iloc[0]["datetime"]}')
    print(f'  Last:  {day_df.iloc[-1]["datetime"]}')
    print(f'  Count: {len(day_df)}')