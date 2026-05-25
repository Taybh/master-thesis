import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__=='__main__':

    df = pd.read_csv('/home/tayebeh/PycharmProjects/LC/EVAL-one4all_SN_result_SN_LC_03_2R_2ndPart_/SN_LC_03_2R_2ndPart__DSC.csv',sep=',')
    saved_column =df.DSC_right #f['dsc_all'] #you can also use


    #lc_performances = pd.read_excel('/home/tayebeh/PycharmProjects/LC/EVAL-one4all_SN_result_SN_LC_03_2R_2ndPart_/SN_LC_03_2R_2ndPart__DSC.csv')
    sns.set(font_scale=2.7)
    sns.set_style('whitegrid')
    ax = sns.violinplot(x=saved_column)
    #ax = sns.violinplot( x='automated', y="Dice", data=saved_column, split=False)
    #ax.set_ylim((0, 1))
    ax.set_xticks((0.6,0.8,0.9, 1.0))
    plt.show()
