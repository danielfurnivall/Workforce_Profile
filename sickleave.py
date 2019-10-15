import pandas as pd
df = pd.read_csv('W:/MFT/Workforce Profiles/sickabs.csv')
#TODO replace file xls with a new csv extract (this was extracted from an xls with multiple sheets)

df['AbsRecordCount'] = df['Pay No'].map(df['Pay No'].value_counts())
df.to_csv('W:/MFT/Workforce Profiles/absworkfile.csv')
