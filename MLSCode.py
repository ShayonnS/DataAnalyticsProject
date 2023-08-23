import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define the file path
file_path = r'file_path'

# Read the CSV file
df = pd.read_csv(file_path)

# Drop rows where 'Weekly_Wages' is empty
df = df.dropna(subset=['WeeklyWage'])

# Check the first few rows to make sure it's loaded correctly
print(df.head())

# calculate weekly wage
team_avg_w_w = df.groupby('Squad')['WeeklyWage'].mean().reset_index()

# Sort the data in descending order
team_avg_w_w = team_avg_w_w.sort_values(by='WeeklyWage', ascending=False)

# Set the style
sns.set_theme(style="whitegrid")

# Set figure size
plt.figure(figsize=(15, 10))

# Plotting with black bars
bars = sns.barplot(data=team_avg_w_w, x='Squad', y='WeeklyWage', color="black", dodge=False)

# Adding a best fit line
positions = np.arange(len(team_avg_w_w))
slope, intercept = np.polyfit(positions, team_avg_w_w['WeeklyWage'], 1)
plt.plot(positions, slope*positions + intercept, color="red", alpha=0.5)  # making the line red for contrast

# Annotating with the position and conference of the team
for p, bar in enumerate(bars.patches):
    height = bar.get_height()
    squad_name = team_avg_w_w.iloc[p]['Squad']
    position = df[df['Squad'] == squad_name]['Postion'].unique()[0]
    conference = df[df['Squad'] == squad_name]['Conference'].unique()[0]
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{position} ({conference})', 
             ha='center', va='bottom', rotation=45, fontsize=9, color='blue')  # Adjust fontsize/color as needed

# Customizing the plot
plt.xticks(rotation=45)
plt.title('Average Weekly Wages by Squad in 2022 with Best Fit Line')
plt.ylabel('Average Weekly Wage in US Dollars')
plt.xlabel('Squad')
plt.tight_layout()

# Show the plot
plt.show()
## fb ref playing time
#to check for multicolinearity
print(df[['Min', 'WeeklyWage', 'Age','xG+/-',]].corr())

## in the matrix Age seems the best on to use as a control b/c it captures minutes relatively well
## age doesn't capture xG+/-, so that is also a good control

#more cleaning
# Check for missing values in the specific columns
print(df[['WeeklyWage', 'Age', 'xG+/-']].isnull().sum())

# Drop rows with missing values (alternative: fill them with a specific value)
df = df.dropna(subset=['WeeklyWage', 'Age', 'xG+/-'])

# Check for infinite values
print((df[['WeeklyWage', 'Age', 'xG+/-']] == float('inf')).sum())

# Replace infinite values with NaN, then drop them (or replace them with a specific value)
df.replace([float('inf'), -float('inf')], np.nan, inplace=True)
df = df.dropna(subset=['WeeklyWage', 'Age', 'xG+/-'])

# Define dependent and independent variables
X = df[['WeeklyWage', 'Age','xG+/-']]  # Independent variable(s)
y = df['Postion']  # Dependent variable

# Add a constant to the independent variable(s)
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# Display the summary
print(model.summary())

