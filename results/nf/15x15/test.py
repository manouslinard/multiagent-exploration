import pandas as pd
import glob
import os

xlsx_files = glob.glob(f'all/*.xlsx')

dfs = []

algo_name = "nf"  # default

for file in xlsx_files:
    df = pd.read_excel(file)
    dfs.append(df)
    file_prefix = file.split('all/')[1].split('_all_')[0]
    if not file_prefix.startswith('p_'):
       algo_name = file_prefix


if len(dfs) == 0:
    raise ValueError("Xlsx files not found")

df_all = pd.concat(dfs, ignore_index=True)

for file in xlsx_files: # removes the old "all" files.
    os.remove(file)

df_all.to_excel(f'all/{algo_name}_all_15x15_True_2.xlsx', index=False)

print(f"DEBUG: Experiments per agent in the combined file (obs_prob = 0.85) = {len(df_all[df_all['Obs_Prob'] == 0.85]) // 6}")

# df_all = pd.read_excel('all\\new_cu_all_15x15_True_2.xlsx')

num_test = len(df_all[df_all['Obs_Prob'] == 0.85]) // 6

df_85 = df_all[df_all['Obs_Prob'] == 0.85]
df_15 = df_all[df_all['Obs_Prob'] == 0.15]

std_total_rounds_85 = df_85.groupby('#_Agents')['Total_Rounds'].std().rename('Std_Total_Rounds')
std_total_rounds_15 = df_15.groupby('#_Agents')['Total_Rounds'].std().rename('Std_Total_Rounds')

selected_columns = df_15.columns.drop(['#_Agents', 'Test'])
df_15_mean = df_15.groupby('#_Agents')[selected_columns].mean()
df_85_mean = df_85.groupby('#_Agents')[selected_columns].mean()

df_15_mean = df_15_mean.merge(std_total_rounds_15, left_index=True, right_index=True, how='left')
df_85_mean = df_85_mean.merge(std_total_rounds_85, left_index=True, right_index=True, how='left')

df_15_mean = df_15_mean.rename(columns={'Total_Rounds': 'Avg_Total_Rounds'})
df_85_mean = df_85_mean.rename(columns={'Total_Rounds': 'Avg_Total_Rounds'})

df_15_mean = df_15_mean.reset_index()
df_85_mean = df_85_mean.reset_index()

concatenated_df = pd.concat([df_15_mean, df_85_mean], ignore_index=True)
concatenated_df = concatenated_df.sort_values(by=['#_Agents', 'Obs_Prob'])

# Count the number of rows in the concatenated DataFrame
num_rows = len(concatenated_df)
print("Number of rows in concatenated DataFrame:", num_rows)

concatenated_df.to_excel(f'{algo_name}_15x15_True_2_{num_test}.xlsx', index=False)
