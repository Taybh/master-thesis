import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__=='__main__':
    x_list = []
    y_list = []
    num_samples = [22,42,62,82]
    origin = 'lc'
    target ='R3'
    #origin = ['lc','sn']
    #target = ['lc']
    for num in num_samples:
        #for org in origin:
        #    if org =='lc':
        #        target = 'R3'
         #   else:
         #       target = 'SN'
        file = './EVAL_scratch' + str(num) + '_' + origin +'_to_' + target+'/unet_'+ origin+ '_Scratch_DSC.csv'
        print(file)
        df = pd.read_csv(file,sep=',')
        #saved_column =df.DSC_right #f['dsc_all'] #you can also use
        mean = df['DSC_right'].mean()
        print(mean)
        x_list.append(num)
        y_list.append(mean)
    print(x_list)
    print(y_list)
    plt.scatter(x_list,y_list)
    plt.xlim(0, 100)
    plt.ylim(0, 1)

    plt.show()


'''
    #lc_performances = pd.read_excel('/home/tayebeh/PycharmProjects/LC/EVAL-one4all_SN_result_SN_LC_03_2R_2ndPart_/SN_LC_03_2R_2ndPart__DSC.csv')
    sns.set(font_scale=2.7)
    sns.set_style('whitegrid')
    ax = sns.violinplot(x=saved_column)
    #ax = sns.violinplot( x='automated', y="Dice", data=saved_column, split=False)
    #ax.set_ylim((0, 1))
    ax.set_xticks((0.6,0.8,0.9, 1.0))
    plt.show()
'''