import matplotlib.pyplot as plt
import pandas as pd
from math import pi


import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# df = pd.DataFrame({
#     'group': ['Baseline', 'LTN', "LTN_B", "LTN_A", "LTN_AB", "LTN_BC", "LNT_AC", "LTN_ABC"],
#     'Accuracy': [79.42, 79.27, 91.28, 78.98, 83.86, 85.35, 80.79, 88.53],
#     'F1': [71.53, 71.26, 90.21, 72.47, 82.74, 84.28, 75.25, 87.13],
#     'Precision': [83.91, 83.44, 89.66, 81.80, 81.92, 83.78, 81.31, 86.59],
#     'Recall': [69.55, 69.29, 90.85, 71.00, 85.44, 86.26, 73.51, 87.83],
#     #'Compliance': [39.94, 36.03, 91.90, 43.79, 94.42, 92.24, 49.22, 90.96],
# })

# df = pd.DataFrame({
#     'group': ['Baseline', 'LTN', "LTN_B", "LTN_A", "LTN_AB", "LTN_BC", "LNT_AC", "LTN_ABC"],
#     'Accuracy': [54.98,57.40,65.68,51.10,57.18,64.67,50.52,58.54],
#     'F1': [54.58,55.79,62.74,50.12,54.72,61.52,49.58,56.39],
#     'Precision': [55.34,55.95,65.02,50.14,55.44,63.83,49.64,57.24],
#     'Recall': [55.36,55.80,62.85,50.14,54.90,61.71,49.61,56.50],
#     #'Compliance': [45.19,50.41,98.98,55.66,100,99.27,51.16,100],
# })

# traffic fines
df = pd.DataFrame({
    'group': ['Baseline', 'LTN', "LTN_B", "LTN_A", "LTN_AB", "LTN_BC", "LNT_AC", "LTN_ABC"],
    'Accuracy': [79.37,77.18,80.16,79.28,80.14,80.13,79.72,79.25],
    'F1': [79.37,77.01,80.15,79.28,80.13,80.13,79.72,79.21],
    'Precision': [79.99,76.95,80.78,79.89,80.80,80.76,80.87,79.41],
    'Recall': [80.10,77.14,80.93,80.04,80.94,80.91,80.73,79.72],
    #'Compliance': [35.18,34.99,99.81,34.79,100,100,34.41,100],
})

# #bpi17
# df = pd.DataFrame({
#     'group': ['Baseline', 'LTN', "LTN_B", "LTN_A", "LTN_AB", "LTN_BC", "LNT_AC", "LTN_ABC"],
#     'Accuracy': [67.61,77.80,81.87,77.42,82.11,81.45,78.44,81.33],
#     'F1': [67.60,75.44,78.95,75.77,79.31,78.70,76.04,78.95],
#     'Precision': [71.84,76.37,83.01,75.77,83.12,81.92,77.18,81.02],
#     'Recall': [72.28,74.86,77.45,75.86,77.84,77.37,75.37,77.91],
# })


# -- Part 1: Create background
categories = list(df)[1:]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Set figure and axis
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# X-axis labels
plt.xticks(angles[:-1], categories, fontsize=10, fontweight='bold')

# Y-axis - Focus on the higher range (70-100) for better distinction
ax.set_rlabel_position(0)
# traffic fines
plt.yticks([76,78, 80, 82], 
           ["76","78", "80", "82"], 
           color="black", size=10)
plt.ylim(76, 82)  # Set minimum to 70 to focus on higher values
#bpi17
# plt.yticks([65, 70, 75, 80, 85], 
#            ["65", "70", "75", "80", "85"], 
#            color="grey", size=8)
# plt.ylim(65, 85)  # Set minimum to 70 to focus on higher values

# Add gridlines for better readability
ax.grid(True, linestyle='-', alpha=0.7)

# Color palette for better differentiation
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
# Plot each group with distinct colors and line styles
for i, (idx, row) in enumerate(df.iterrows()):
    values = row.drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', 
            label=row['group'], color=colors[i % len(colors)])
    ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

# Add a second subplot for Compliance specifically (since it has lower values)
ax2 = fig.add_subplot(2, 3, 6)  # Add a subplot in the bottom right
df = pd.DataFrame({
    'group': ['Baseline', 'LTN', "LTN_B", "LTN_A", "LTN_AB", "LTN_BC", "LNT_AC", "LTN_ABC"],
    'Accuracy': [79.37,77.18,80.16,79.28,80.14,80.13,79.72,79.25],
    'F1': [79.37,77.01,80.15,79.28,80.13,80.13,79.72,79.21],
    'Precision': [79.99,76.95,80.78,79.89,80.80,80.76,80.87,79.41],
    'Recall': [80.10,77.14,80.93,80.04,80.94,80.91,80.73,79.72],
    'Compliance': [35.18,34.99,99.81,34.79,100,100,34.41,100],
})

groups = df['group']
compliance_values = df['Compliance']

# Plot Compliance as a bar chart
bars = ax2.bar(groups, compliance_values, color=colors)
# ax2.set_title('Compliance Scores', fontsize=10)
ax2.set_ylim(0, 100)
ax2.set_xlabel('Compliance', fontweight='bold')
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_ylabel('Score')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.grid(axis='y', linestyle='', alpha=0.7)

# Adjust subplot size
pos = ax2.get_position()
ax2.set_position([pos.x0, pos.y0 + 0.15, pos.width, pos.height * 0.85])

# Add legend with a better position and formatting
plt.figlegend(loc='upper right',
          ncol=2, fancybox=True, shadow=True, fontsize=14)

# Add title and adjust layout
plt.tight_layout()

# Save and show
plt.savefig('plots/trafficfines.pdf', dpi=600, bbox_inches='tight')
plt.savefig('plots/trafficfines.png', dpi=1000, bbox_inches='tight')
plt.show()