from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('../data/gz2_hart16_classes_simple.csv')

df.hist('total_classifications')
plt.xlabel('# of classifications per galaxy')
plt.ylabel('# of galaxies')
plt.yscale('log')
plt.show()

df.groupby('hubble_class').dr7objid.count().plot(kind='bar')
plt.xlabel('Hubble class')
plt.ylabel('# of galaxies')
plt.show()

print('Total: ', df.gz2_class.count())
print('\n')
print('Per Hubble class: ', df.groupby('hubble_class').dr7objid.count())
print('\n')
print('Per gz2 class: ', df.groupby('gz2_class').dr7objid.count())
