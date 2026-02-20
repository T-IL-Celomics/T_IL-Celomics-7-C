import pandas as pd
xls = pd.ExcelFile('Gab_Normalized_Combined_C7.xlsx')
print(xls.sheet_names)