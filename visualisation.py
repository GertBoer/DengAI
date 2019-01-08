import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/dengue_labels_train.csv")

df = df.groupby(['year', 'city'])['total_cases'].sum()

df.plot.bar()
plt.title('Total cases per year')
plt.xticks(rotation=45)
plt.show()
