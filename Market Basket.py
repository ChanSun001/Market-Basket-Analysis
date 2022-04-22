import os
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

cwd = os.getcwd()
print("Previous working directory: {0}".format(cwd))
os.chdir('/Users/jsun/Documents/R Work Files/Python')
cwd = os.getcwd()
print(("Current working directory: {0}".format(cwd)))


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


df = pd.read_excel('jonathan.sun_basket_data.xlsx')
print(df.head())

df[df["qty"] < 0] = 0


basket = (df.groupby(['custNo', 'product'])['qty']
          .sum().unstack().reset_index().fillna(0)
          .set_index('custNo'))

basket.to_csv("file_name.csv")
#basket = basket.drop(columns = basket.columns[0])
#basket = basket.drop(0)

basket_sets = basket.applymap(encode_units)

frequent_item_sets = apriori(basket_sets, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_item_sets, metric="lift", min_threshold=1)

#basket_sets.to_csv("basket_set.csv")
rules.to_csv("python_rules.csv")

#print(basket_sets['EMTCON112'].sum())
#print(basket_sets['EMTCON1'].sum())


class FileSettings(object):
    def __init__(self, file_name_split, row_size=100):
        self.file_name_split = file_name_split
        self.row_size = row_size


class FileSplitter(object):

    def __init__(self, file_settings):
        self.file_settings = file_settings

        if type(self.file_settings).__name__ != "FileSettings":
            raise Exception("Please pass correct instance ")

        self.data = pd.read_csv(self.file_settings.file_name_split,
                              chunksize=self.file_settings.row_size)

    def run(self, directory="/Users/jsun/Documents/R Work Files/Python"):

        try:
            os.makedirs(directory)
        except Exception as e:
            pass

        counter = 0

        while True:
            try:
                file_name_split = "{}/{}_{}_row_{}.csv".format(
                    directory, self.file_settings.file_name_split.split(".")[0], counter, self.file_settings.row_size
                )
                data = next(self.data).to_csv(file_name_split)
                counter = counter + 1
            except StopIteration:
                break
            except Exception as e:
                print("Error:", e)
                break

        return True


def main():
    helper = FileSplitter(FileSettings(
        file_name_split='python_rules.csv',
        row_size=700000
    ))
    helper.run()


main()
