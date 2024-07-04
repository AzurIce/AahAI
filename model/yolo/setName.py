import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('table/image_data.csv')

# 去重
df_unique = df.drop_duplicates()
df_unique.to_csv('table/image_data_unique.csv', index=False)
# df = pd.read_csv('table/image_data_unique.csv')
# print(df_unique)
#
with open('output.txt', 'w') as f:
    # Iterate over each row in the DataFrame
    for index, row in df_unique.iterrows():
        # Extract id and name from the current row
        index_value = row['index']
        name_value = row['name']

        # Format the output as "<id>: <name>"
        line = f"{index_value}: {name_value}\n"

        # Write the formatted line to the file
        f.write(line)
