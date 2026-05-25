import csv
import statistics
import pandas as pd

###check the convergence criterion
path = "logs/scratch/lc/csv/loss_val.csv"
#file = path + ""
#print(file)
#df = pd.read_csv(path, usecols=[1])

for i in range(5):
    data = pd.read_csv(path, header = 0, usecols = [1])#["unet_LC_Scratchf1_wdh1"]
    print(data)
    print(data.mean())


    for index, value in enumerate(df['']):
        print(value)
    for index, value in enumerate(df['unet_LC_Scratchf1_wdh1']):
        print('index:' + str(index) + '  value:' + str(value))
'''
#saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
for index, value in enumerate(df1['unet_LC_Scratchf1']):
    print('index:'+ str(index) + '  value:'+ str(value))
    if value ==0:
        value_fold0.append(df1[dice][index])
        print(df1[dice][index])
    elif value ==1:
        value_fold1.append(df1[dice][index])
        print(df1[dice][index])
    elif value ==2:
        value_fold2.append(df1[dice][index])
        print(df1[dice][index])
    elif value ==3:
        value_fold3.append(df1[dice][index])
        print(df1[dice][index])
    else:
        value_fold4.append(df1[dice][index])
        print(df1[dice][index])
'''

#with open('logs/scratch/lc/csv/loss_val.csv') as f:
  #  cf = csv.DictReader(f, fieldnames=['city'])
 #   for row in cf:
  #      print row['city']


with open(path, "r") as csv_read:
    r = csv.reader(csv_read, delimiter = ",")
    #next(r, None) #Skip headers first row
    #for column in r:
        #print (column[0])

###1st senario
    #for row in r:
    #    if row !=0:
     #       print(row)
    for row in r:
        if row !=0:
            for column in row:
                print(row)
                print(column)
                val_verylast = []
                val_last=[]
                #min_last = 0
                #min_verylast = []

                val_last.append(row[190:220])
                val_verylast.append(row[220:250])
                min_verylast = min(val_verylast)
                min_last = min(val_last)
                print(min_last)
                if (min_verylast) < (min_last *0.1):
                    print('no convergence')
                else:
                    print('convergance')

###

#df = pd.read_csv(file, sep=',')
#for column in df:
 #   print(column[2])

# saved_column =df.DSC_right #f['dsc_all'] #you can also use
#mean = df['DSC_right'].mean()
#print(mean)

'''

###check the convergence criterion
import numpy as np
import pandas as pd


def trendline(data, order=1):
    coeffs = np.polyfit(data.index.values, list(data), order)
    slope = coeffs[-2]
    return float(slope)


# Sample Dataframe
revenue = [0.85, 0.99, 1.01, 1.12, 1.25, 1.36, 1.28, 1.44]
year = [1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000]

# check if values are exactly same
if (len(set(revenue))) <= 1:
    print(0)
else:
    df = pd.DataFrame({'year': year, 'revenue': revenue})

    slope = trendline(df['revenue'])
    print(slope)



for words in cleaned_example:
    yes = 0
    no = 0
    for word1, word2 in zip(words, words[1:]):
        if sentiment(word1)[0] < sentiment(word2)[0]:
            yes = yes + 1
        else:
            no = no + 1
    if yes > no:
        print "yes"
    elif yes < no:
        print "no"
    else:
        print "`yes` = `no`"
        
'''