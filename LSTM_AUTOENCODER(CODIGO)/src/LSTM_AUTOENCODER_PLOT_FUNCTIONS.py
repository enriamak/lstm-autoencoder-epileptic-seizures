import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_sequence(sample1,sample2, i_from=None, i_to=None):
    if i_from != None and i_to!=None:
        sample1_data=sample1[0].iloc[i_from:i_to].values
        sample2_data=sample2[0].iloc[i_from:i_to].values
    else:
        sample1_data=sample1[0].values
        sample2_data=sample2[0].values

    sample1_label=sample1[1]
    sample2_label=sample2[1]
    plt.style.use('dark_background')
    plt.figure(figsize=(16, 10))
    line1, = plt.plot(range(len(sample1_data)), sample1_data, label=str(sample1_label))
    line2, = plt.plot(range(len(sample2_data)), sample2_data, label=str(sample2_label),color="r")
    plt.legend(handles=[line1, line2], loc='best')
    plt.grid(b=True, color='aqua', alpha=0.3, linestyle='dashdot')
    plt.title('Ventana')
    #plt.savefig('difference_waves.png',bbox_inches='tight')
    plt.show

def plot_multiples_windows(sample1,sample2):#exmp sample1=5x120
    plt.figure(dpi=300) 
    
    sample1_data=sample1.flatten()
    sample2_data=sample2.flatten()

    plt.style.use('dark_background')
    plt.figure(figsize=(16, 10))
    line1, = plt.plot(range(len(sample1_data)), sample1_data, label=str(0))
    line2, = plt.plot(range(len(sample2_data)), sample2_data, label=str(1),color="r")
    plt.legend(handles=[line1, line2], loc='best')
    plt.grid(b=True, color='aqua', alpha=0.3, linestyle='dashdot')
    plt.title('Ventana')
    #plt.savefig('difference_waves.png',bbox_inches='tight')
    plt.show

def plot_loss_throw_epochs(history,MODEL_NAME):
    plt.style.use('dark_background')
    ax=plt.figure(dpi=300).gca()
    ax.plot(history['train'])
    ax.plot(history['val'],color="r")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train','test'])
    plt.grid(color='aqua', alpha=0.1, linestyle='dashdot')
    plt.title(str('Loss over epochs '+MODEL_NAME))
    plt.savefig(str(MODEL_NAME+'.png'),bbox_inches='tight')
    plt.show()
    
def plot_range_loss(losses_normal,losses_seizures=None, MODEL_NAME='model',THRESHOLD=None):
    plt.figure(dpi=300) 
    plt.style.use('dark_background')
    sns.histplot(losses_normal, bins=30, kde=True, stat="density", linewidth=0)
    plt.axvline(THRESHOLD, color='y', linestyle='dashed')
    plt.grid(color='aqua', alpha=0.1, linestyle='dashdot')
    if losses_seizures!=None:
        plt.title(str('Threshold by Loss '+MODEL_NAME))
        losses_seizures=[x for x in losses_seizures if x <= 5000]
        sns.histplot(losses_seizures, bins=30, kde=True,color="r", stat="density", linewidth=0)
        plt.savefig(str(MODEL_NAME+'_loss_normal_anomaly.png'))
    else:
        plt.title(str('Loss Comparison '+MODEL_NAME))
        plt.savefig(str(MODEL_NAME+'_loss_normal_train.png'),bbox_inches='tight')
    plt.show()
    
    
def plot_confusion_matrix(cf_matrix,MODEL_NAME):
    plt.figure(dpi=300) 
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    
    labels = np.asarray(labels).reshape(2,2)
    
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    
    ax.set_title(str('CM '+MODEL_NAME));
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    
    ## Display the visualization of the Confusion Matrix.
    plt.savefig(str(MODEL_NAME+'_confusion_matrix.png'),bbox_inches='tight')
    plt.show()
    
    
def plot_clasification_report(clf_report, MODEL_NAME):
    plt.figure(dpi=300) 
    ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True,cmap="Blues",cbar=False)#YlGnBu
    ax.set_title(str('CR '+MODEL_NAME));
    plt.savefig(str(MODEL_NAME+'_clasification_report.png'),bbox_inches='tight')
    plt.show()
    