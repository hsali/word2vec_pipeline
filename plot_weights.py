import pandas as pd
df = pd.read_csv("data_document_scores/word_document_dispersion_weights.csv").set_index("words")
df = df.sort_values('alpha')


import seaborn as sns
import pylab as plt
plt.plot(df.alpha.values)
#plt.ylim(0,1.5)
#plt.xlim(0,1000)
plt.show()


