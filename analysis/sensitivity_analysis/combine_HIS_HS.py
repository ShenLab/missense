import sys
import pandas as pd

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])

df1 = df1[df1['gnomad_exome'] < 1e-6]
df2 = df2[df2['gnomad_exome'] < 1e-4]
df = pd.concat([df1, df2], axis=0)

df.to_csv(sys.argv[3], index=False)
