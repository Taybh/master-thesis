import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
import numpy as np
import math
import statistics
import plotly.graph_objects as go


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

if __name__=='__main__':
    #x_list = []
    #y_22 = []
    #y_42 = []
    #num_samples = [22,42] #,62,82
    #origin = ['lc','sn']
    #target = ['lc']
    dice ='DSC_all'
    num = 10
    target = 'SnToLc' #  'SnToLc' 'LcToSn'
    if target == 'SnToLc':
        scratch_target = '_lc_to_R3/unet_lc_Scratch_DSC.csv'

    else:
        scratch_target = '_sn_to_SN/unet_sn_Scratch_DSC.csv'

    file2 = 'EVAL_diff_seed_scratch' + str(num) + scratch_target
    type = ['decay','freeze']#
    pattern = ['last', 'up2', 'up1', 'up0', 'down3', 'down2', 'down1', 'whole']
    #colors = ['black', 'dimgray', 'darkgrey','rosybrown','darkred','red','olive',
     #         'yellow','olivedrab','lawngreen','dodgerblue','steelblue','aqua',
      #        'violet','purple','chocolate']

    #labels = ['22', '42']

    fig, ax = plt.subplots()

    #fig = go.Figure()
    #width = 0.35
    #x = np.arange(len(labels))
    x = np.arange(13)#17,9,13
    #for num in num_samples:
    #counter =0

    #result = []

    means = []
    maxs=[]
    mins =[]
    stds =[]
    for typ in type:
        if typ == 'freeze':
            pattern = ['down3', 'down2', 'down1', 'whole']
        for patt in pattern:
            plot_data = []
            value_fold0 = []
            value_fold1 = []
            value_fold2 = []
            value_fold3 = []
            value_fold4 = []
            counter=0
            #for org in origin:
            #for tar in target:
            #file = '/home/tayebeh/PycharmProjects/LC/LCSN-Seg-f4f50f3/EVAL_scratch' + str(num) + '_lc_to_R3/unet_lc_Scratch_DSC.csv'

            #with open('serials.csv') as csvfile:
             #   reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
             #   for row in reader:
              #      if row[0].strip() == key:
               #         result.append(row[1].strip())


            file1 = 'EVAL_diff_seed_TL' + str(num) + '_'+ typ +'_'+patt + os.sep +target+'_DSC.csv'
            #file2 = 'EVAL_TL' + str(num) + '_' + 'freeze' + '_' + patt + os.sep + target + '_DSC.csv'
            print(file1)
            #print(file2)
            #with open(file1) as csvfile:
             #   reader = csv.reader(csvfile, delimiter=',')
              #  for column in reader:
               #     if column[1]==0:
                #        result.append()

            df1 = pd.read_csv(file1,sep=',')
            #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
            for index, value in enumerate(df1['fold']):
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

            plot_data.append(statistics.mean(value_fold0))
            plot_data.append(statistics.mean(value_fold1))
            plot_data.append(statistics.mean(value_fold2))
            plot_data.append(statistics.mean(value_fold3))
            plot_data.append(statistics.mean(value_fold4))



            means.append(statistics.mean(plot_data))
            maxs.append(max(plot_data))
            mins.append(min(plot_data))
            stds.append(statistics.stdev(plot_data))

            #mean1 = df1[dice].mean()
            #y_22.append(mean1)


            #plt.bar(num, mean, color = colors[counter], label=typ+'_'+patt)

            #ax.axis('equal')
            #plt.legend()
            #name = target + '_TL_decay_' + dice + '.png'
            #plt.xlim(0, 100)
            #plt.ylim(0, 1)

    df2 = pd.read_csv(file2, sep=',')
    value_fold0 = []
    value_fold1 = []
    value_fold2 = []
    value_fold3 = []
    value_fold4 = []
    plot_data = []
    for index, value in enumerate(df2['fold']):
        print('index:' + str(index) + '  value:' + str(value))
        if value == 0:
            value_fold0.append(df2[dice][index])
            print(df2[dice][index])
        elif value == 1:
            value_fold1.append(df2[dice][index])
            print(df2[dice][index])
        elif value == 2:
            value_fold2.append(df2[dice][index])
            print(df2[dice][index])
        elif value == 3:
            value_fold3.append(df2[dice][index])
            print(df2[dice][index])
        else:
            value_fold4.append(df2[dice][index])
            print(df2[dice][index])

    plot_data.append(statistics.mean(value_fold0))
    plot_data.append(statistics.mean(value_fold1))
    plot_data.append(statistics.mean(value_fold2))
    plot_data.append(statistics.mean(value_fold3))
    plot_data.append(statistics.mean(value_fold4))

    means.append(statistics.mean(plot_data))
    maxs.append(max(plot_data))
    mins.append(min(plot_data))
    stds.append(statistics.stdev(plot_data))
    labels = ['decay_last', 'dacay_up2', 'decay_up1', 'decay_up0', 'decay_down3',
                   'decay_down2','decay_down1', 'decay_whole', 'freeze_down3',
              'freeze_down2', 'freeze_down1', 'freeze_whole','scratch'] #'freeze_last', 'freeze_up2', 'freeze_up1', 'freeze_up0',
                                                                 #'freeze_down3', 'freeze_down2', 'freeze_down1', 'freeze_whole',


    #plt.bar(x, y_22)#,color=colors)
    # create stacked errorbars:
    print(len(plot_data))
    print(len(means))
    print(len(stds))
    print(len(mins))
    print(len(maxs))

    quart = []

    zip_lower = zip(means, mins)
    zip_upper = zip(maxs,means)
    for list1_i, list2_i in zip_lower:
        quart.append(list1_i - list2_i)

    for list1_i, list2_i in zip_upper:
        quart.append(list1_i - list2_i)

    #print(len(quart))
    plt.errorbar(x, means, stds, mfc='red',
              fmt='ok', lw=3, label='mean and std')
    #plt.errorbar(x, means, [quart[0:9],quart[9:18]],  #17,34 , 9,18
    #             fmt='.k', ecolor='gray', lw=1,label='min and max of values')
    #print(list(set(maxs) - set(means)))

    plt.xticks(x, ('decay_last', 'dacay_up2', 'decay_up1', 'decay_up0', 'decay_down3',
                   'decay_down2','decay_down1', 'decay_whole', 'freeze_down3',
              'freeze_down2', 'freeze_down1', 'freeze_whole','scratch'),rotation='vertical') #'freeze_last', 'freeze_up2', 'freeze_up1', 'freeze_up0',
                      #'freeze_down3', 'freeze_down2', 'freeze_down1', 'freeze_whole'
    ax.legend()
    ax.set_ylabel('Dice Similarity Coefficient')
    if dice == 'DSC_right':
        temp = 'Right'
    elif dice == 'DSC_left':
        temp = 'Left'
    else:
        temp = 'Combined'

    if num ==10:
        temp_num = 7
    elif num == 20:
        temp_num = 14
    else:
        temp_num = num

    if target == 'LcToSn':
        title = temp + ' SNpc Segmentation on ' + str(temp_num) + ' Target Size'  #+ '_TL_'
    else:
        title = temp + ' LC Segmentation on ' + str(temp_num) + ' Target Size'   # + '_TL_'


    name = 'diff_seed_' + str(num)+'_target_size_' + target  +'_'+ dice  + '.png' #+ '_TL_'


    #ax.set_title(title)
    ax.set_xlabel('Learning Method')
    for index, value in enumerate(means):
        plt.text(index+0.2, value+0.02,(str(truncate(value,4))),color='red',rotation='vertical')
    #ax.text(0, 0.8, 'mean '+'\n'+'values', fontsize=10, color='red',rotation ='vertical')
    for index, value in enumerate(stds):
        plt.text(index-0.4, means[index]+stds[index]+0.02,(''+
                               str(truncate(value,4))),rotation='vertical') #,color='red'
    plt.ylim(0, 1.1)
    plt.savefig(name,bbox_inches='tight')
    plt.show()


    #plt.savefig(name)

    #ax.set_ylabel('Dcs')
    #ax.set_title('Dice for LC segmentation')
    #ax.set_xticks(x)
    #ax.set_xticklabels(labels)



    #fig.tight_layout()
    #plt.show()
    #plt.savefig(name)











'''
###############
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


'''