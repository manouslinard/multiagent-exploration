import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


df = pd.read_excel('astar_swarm.xlsx')

# weights sum up to 0.
weights = {
    'Coverage': 0.9,
    'Finished_Agents': 0.7,
    'Experiment_Time': -0.8,
    'Episode_Time': -0.3,
    'Agent_Finish_Time': -0.5,
}

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[['Coverage', 'Finished_Agents', 'Experiment_Time', 'Episode_Time', 'Agent_Finish_Time']]),
                              columns=['Coverage', 'Finished_Agents', 'Experiment_Time', 'Episode_Time', 'Agent_Finish_Time'])

df['Composite_Score'] = (df_normalized['Coverage'] * weights['Coverage'] +
                         df_normalized['Finished_Agents'] * weights['Finished_Agents'] +
                         df_normalized['Experiment_Time'] * weights['Experiment_Time'] +
                         df_normalized['Episode_Time'] * weights['Episode_Time'] +
                         df_normalized['Agent_Finish_Time'] * weights['Agent_Finish_Time'])

# print(df)

avg_scores = df.groupby('#_Agents')['Composite_Score'].mean().reset_index(name='Average_Score')
print(avg_scores)

plt.plot(avg_scores['#_Agents'], avg_scores['Average_Score'], marker='o', linestyle='-', color='b')
plt.xlabel('# of Agents')
plt.xticks(avg_scores['#_Agents'].astype(int))
plt.ylabel('Average Score')
plt.grid(True)
plt.show()
