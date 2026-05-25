import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
'''
##### left Dice LC: dd1, fd1, fd2, scratch, decay whole
dice ='DSC_left'
fig, ax = plt.subplots()
x = [5,7,14,81]
nums = [5,10,20,82]
mean_scratch = []
std_scratch =[]
mean_dd1= []
std_dd1 =[]
mean_fd1= []
std_fd1 =[]
mean_fd2 = []
std_fd2 =[]
mean_dw = []
std_dw =[]
for num in nums:
    plot_data = []
    file = 'EVAL_scratch' + str(num) + '_lc_to_R3/unet_lc_Scratch_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_scratch.append(statistics.mean(plot_data))
    std_scratch.append(statistics.stdev(plot_data))

####dd1####
###########
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_' + 'decay_down1/SnToLc_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_dd1.append(statistics.mean(plot_data))
    std_dd1.append(statistics.stdev(plot_data))

#####fd1####
############
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_freeze_down1/SnToLc_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_fd1.append(statistics.mean(plot_data))
    std_fd1.append(statistics.stdev(plot_data))

####fd2######
############
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_freeze_down2/SnToLc_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_fd2.append(statistics.mean(plot_data))
    std_fd2.append(statistics.stdev(plot_data))


#####decay whole####
###################
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_decay_whole/SnToLc_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_dw.append(statistics.mean(plot_data))
    std_dw.append(statistics.stdev(plot_data))

plt.plot(x, mean_scratch ,  color= 'r' ,label='scratch')
plt.plot(x, mean_dd1 , color= 'g', label='decay down1')
plt.plot(x, mean_fd1 ,color= 'b', label='freeze down1')
plt.plot(x, mean_fd2 ,color= 'y', label='freeze down2')
plt.plot(x, mean_dw ,color= 'c', label='decay whole')


ax.legend()
ax.set_ylabel('Dice Similarity Coefficient')
name =  dice  + '_LC.png' #+ '_TL_'
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Dice Similarity Coefficient')
plt.ylim(0, 1.5)
plt.savefig(name,bbox_inches='tight')
plt.show()



############################################
##### right Dice LC: fd1, dd1, scratch, decay last
dice ='DSC_right'
fig, ax = plt.subplots()
x = [5,7,14,81]
nums = [5,10,20,82]
mean_scratch = []
std_scratch =[]
mean_dd1= []
std_dd1 =[]
mean_fd1= []
std_fd1 =[]
mean_dl = []
std_dl =[]
for num in nums:
    plot_data = []
    file = 'EVAL_scratch' + str(num) + '_lc_to_R3/unet_lc_Scratch_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_scratch.append(statistics.mean(plot_data))
    std_scratch.append(statistics.stdev(plot_data))

####dd1####
###########
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_' + 'decay_down1/SnToLc_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_dd1.append(statistics.mean(plot_data))
    std_dd1.append(statistics.stdev(plot_data))

#####fd1####
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_freeze_down1/SnToLc_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_fd1.append(statistics.mean(plot_data))
    std_fd1.append(statistics.stdev(plot_data))

####dl######
############
############
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_decay_last/SnToLc_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_dl.append(statistics.mean(plot_data))
    std_dl.append(statistics.stdev(plot_data))

plt.plot(x, mean_scratch ,  color= 'r' ,label='scratch')
plt.plot(x, mean_dd1 , color= 'g', label='decay down1')
plt.plot(x, mean_fd1 ,color= 'b', label='freeze down1')
plt.plot(x, mean_dl ,color= 'c', label='decay last')


ax.legend()
ax.set_ylabel('Dice Similarity Coefficient')
name =  dice  + '_LC.png' #+ '_TL_'
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Dice Similarity Coefficient')
plt.ylim(0, 1.5)
plt.savefig(name,bbox_inches='tight')
plt.show()
'''
##### left Dice SN: dd1, fd2, decay last  fwhole, dwhole,, scratch,
dice ='DSC_left'

fig, ax = plt.subplots()
x = [5,7,14,81]
nums = [5,10,20,82]
mean_scratch = []
std_scratch =[]
mean_dd1= []
std_dd1 =[]
mean_fd2= []
std_fd2 =[]
mean_dL = []
std_dL =[]
mean_fw = []
std_fw =[]
mean_dw = []
std_dw =[]
for num in nums:
    plot_data = []
    file = 'EVAL_scratch' + str(num) + '_sn_to_SN/unet_sn_Scratch_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_scratch.append(statistics.mean(plot_data))
    std_scratch.append(statistics.stdev(plot_data))

####dd1####
###########
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_' + 'decay_down1/LcToSn_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_dd1.append(statistics.mean(plot_data))
    std_dd1.append(statistics.stdev(plot_data))

#####fd2####
############
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_freeze_down2/LcToSn_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_fd2.append(statistics.mean(plot_data))
    std_fd2.append(statistics.stdev(plot_data))

### decay last######
############
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_decay_last/LcToSn_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_dL.append(statistics.mean(plot_data))
    std_dL.append(statistics.stdev(plot_data))

####fwhole,

for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_freeze_whole/LcToSn_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_fw.append(statistics.mean(plot_data))
    std_fw.append(statistics.stdev(plot_data))



# dwhole

for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_decay_whole/LcToSn_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_dw.append(statistics.mean(plot_data))
    std_dw.append(statistics.stdev(plot_data))


plt.plot(x, mean_scratch ,  color= 'r' ,label='scratch')
plt.plot(x, mean_dd1 , color= 'g', label='decay down1')
plt.plot(x, mean_fd2 ,color= 'b', label='freeze down2')
plt.plot(x, mean_fw ,color= 'm', label='freeze whole')
plt.plot(x, mean_dL ,color= 'y', label='decay last')
plt.plot(x, mean_dw ,color= 'c', label='decay whole')


ax.legend()
ax.set_ylabel('Dice Similarity Coefficient')
name =  dice  + '_SN.png' #+ '_TL_'
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Dice Similarity Coefficient')
plt.ylim(0, 1.5)
plt.savefig(name,bbox_inches='tight')
plt.show()




##### right Dice SN: dd2, dwhole, fwhole,fd2, scratch
dice ='DSC_right'

fig, ax = plt.subplots()
x = [5,7,14,81]
nums = [5,10,20,82]
mean_scratch = []
std_scratch =[]
mean_dd2= []
std_dd2 =[]
mean_fd2= []
std_fd2 =[]
mean_fw = []
std_fw =[]
mean_dw = []
std_dw =[]
for num in nums:
    plot_data = []
    file = 'EVAL_scratch' + str(num) + '_sn_to_SN/unet_sn_Scratch_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_scratch.append(statistics.mean(plot_data))
    std_scratch.append(statistics.stdev(plot_data))
### dd2
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_decay_down2/LcToSn_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_dd2.append(statistics.mean(plot_data))
    std_dd2.append(statistics.stdev(plot_data))

###dwhole
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_decay_whole/LcToSn_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_dw.append(statistics.mean(plot_data))
    std_dw.append(statistics.stdev(plot_data))
###fwhole
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_freeze_whole/LcToSn_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_fw.append(statistics.mean(plot_data))
    std_fw.append(statistics.stdev(plot_data))

###fd2
for num in nums:
    plot_data = []
    file = 'EVAL_TL' + str(num) + '_freeze_down2/LcToSn_DSC.csv'
    df = pd.read_csv(file,sep=',')
    #saved_column1 =df1.DSC_right #f['dsc_all'] #you can also use
    for index, value in enumerate(df[dice]):
        print('index:'+ str(index) + '  value:'+ str(value))
        plot_data.append(value)

    mean_fd2.append(statistics.mean(plot_data))
    std_fd2.append(statistics.stdev(plot_data))

# dd2, dwhole, fwhole,fd2
plt.plot(x, mean_scratch ,  color= 'r' ,label='scratch')
plt.plot(x, mean_dd2 , color= 'g', label='decay down2')
plt.plot(x, mean_fd2 ,color= 'b', label='freeze down2')
plt.plot(x, mean_fw ,color= 'y', label='freeze whole')
plt.plot(x, mean_dw ,color= 'c', label='decay whole')


ax.legend()
ax.set_ylabel('Dice Similarity Coefficient')
name =  dice  + '_SN.png' #+ '_TL_'
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Dice Similarity Coefficient')
plt.ylim(0, 1.5)
plt.savefig(name,bbox_inches='tight')
plt.show()


