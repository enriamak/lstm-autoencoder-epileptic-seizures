
from src.LSTM_AUTOENCODER_CLASES import *
from src.LSTM_AUTOENCODER_DATA_FUNCTIONS import *
from src.LSTM_AUTOENCODER_PLOT_FUNCTIONS  import *


#directory='/home/tfgadmin/enri/input_features/'
directory='C:/Users/enria/Desktop/input_features/input_features/'
#directory=''

x_name_normal='normal_1_0_data_x.npy'
y_name_normal='normal_1_0_data_y.npy'
x_name_seizure='seizure_1_0_data_x.npy'
y_name_seizure='seizure_1_0_data_y.npy'

N_KFOLD=5

PLOT_RESULT=True
#PLOT_RESULT=False

#KFOLD_MODE=True 
KFOLD_MODE= False  #train_test_split

MODEL_ACTION='LOAD'
#MODEL_ACTION='TRAIN'

LIST_CONF_CHANNELS=[3] #1=resta canal 6 y 10   2=concatenacion    3=media aritmetica
LIST_N_EPOCHS=[150]   
LIST_LEARNING_RATE=[1e-3]   
LIST_ENCODING_DIM=[2,7,64]   #nHidden
LIST_SEQL=[2,5]   #numero de ventanas por cada muestra      seqL=5 ==  300000x2x128

LIST_CONF_CHANNELS=[3]
LIST_N_EPOCHS=[150]
LIST_LEARNING_RATE=[1e-3]
LIST_ENCODING_DIM=[7]
LIST_SEQL=[2]

for CONF_CHANNELS in LIST_CONF_CHANNELS:
    for N_EPOCHS in LIST_N_EPOCHS:
        for LEARNING_RATE in LIST_LEARNING_RATE:
            for ENCODING_DIM in LIST_ENCODING_DIM:
                for SEQL in LIST_SEQL:
                    
                    #cargamos datos normales
                    data_x_normal, data_y_normal = load_data(directory,x_name_normal,y_name_normal,conf_channel=CONF_CHANNELS, pandas=False, seql=SEQL)
                    
                    #subsampling
                    data_x_normal=data_x_normal[:10000]
                    data_y_normal=data_y_normal[:10000]
                    
                    if MODEL_ACTION == 'LOAD': #cargamos datos anómalos para testear
                        data_x_seizure, data_y_seizure = load_data(directory,x_name_seizure,y_name_seizure,conf_channel=CONF_CHANNELS, pandas=False, seql=SEQL)
                        data_x_seizure=data_x_seizure[np.unique(np.where(data_y_seizure == 1)[0]).tolist()]
                        data_y_seizure=data_y_seizure[np.unique(np.where(data_y_seizure == 1)[0]).tolist()]
                        test_seizure, _, _ = convert_to_tensor(data_x_seizure)
                    
                    kf = KFold(n_splits=N_KFOLD)
                    
                    for idx,(train_index, test_index) in enumerate(kf.split(data_x_normal)):
                        
                        if KFOLD_MODE:
                            print(f'######### KFOLD: {idx} #########   TRAIN: {len(train_index)} TEST: {len(test_index)} \n')
                            train_normal, test_normal = data_x_normal[train_index], data_x_normal[test_index]
                            
                        else:
                            print(f'Train_test_split...',end=" ")
                            train_normal, test_normal = train_test_split(data_x_normal, test_size=0.20, random_state=42)
                            idx=-1
                            print('DONE')
                       
                        
                        MODEL_NAME='kf_'+str(idx)+'_tc_'+str(CONF_CHANNELS)+'_ep_'+str(N_EPOCHS)+'_sL_'+str(SEQL)+'_eD_'+str(ENCODING_DIM)+'_lr_'+str(LEARNING_RATE)+'_T_'+str(len(train_normal))+'_t_'+str(len(test_normal))
                        
                        
                        print('Creating tensor dataset... ', end=" ")
                        train_normal, seq_len, n_features = convert_to_tensor(train_normal)
                        test_normal, _ , _ = convert_to_tensor(test_normal)
                        print('DONE')
                        
                        
                        print('Creating model... ', end=" ")
                        model = Autoencoder(seq_len, n_features, ENCODING_DIM)
                        model = model.to(model.device)
                        print('DONE \n')
                        
                        
                       
                        if MODEL_ACTION == 'TRAIN':
                            model, history = model.train_model(model, train_normal, test_normal, train_index, test_index, N_EPOCHS, LEARNING_RATE, MODEL_NAME, PLOT_RESULT)
                            print('TRAIN Completed!!\n')
                            
                            
                        elif MODEL_ACTION == 'LOAD':
                            print(f'Loading model...', end=" ")
                            model.load_state_dict(torch.load(str(MODEL_NAME+'.pth'), map_location=model.device))
                            print('DONE \n')
                            
                            #load losses dict
                            with open(str(MODEL_NAME+'.pickle'), 'rb') as handle: history = pickle.load(handle)
                            if PLOT_RESULT: plot_loss_throw_epochs(history,MODEL_NAME)
                            
                            #calculate train losses 
                            _, losses_train = predict(model, train_normal)
                            
                            #calculate threshold
                            THRESHOLD = threshold_otsu(np.array(losses_train))
                            
                            #plot losses and threshold
                            plot_range_loss(losses_train, MODEL_NAME=MODEL_NAME,THRESHOLD=THRESHOLD)
                            
                            begin_time = time.strftime("%H:%M:%S", time.localtime())
                            print(f'Time: {begin_time}', end=" ")
                            #predict test_normal, datos no vistos por el modelo
                            _ , losses_normal = predict(model, test_normal)
                            
                            #igualamos numero muestras 
                            test_seizure = test_seizure[:len(test_normal)]
                            
                            #predict test_seizure, datos no vistos por el modelo
                            _ , losses_seizure = predict(model, test_seizure)
                            
                            #plot histograma ambas clases
                            plot_range_loss(losses_normal,losses_seizure,MODEL_NAME,THRESHOLD)
                            
                            #calculamos metricas
                            correct_anomaly = sum(l > THRESHOLD for l in losses_seizure)
                            correct_normal = sum(l <= THRESHOLD for l in losses_normal)
                            
                            end_time = time.strftime("%H:%M:%S", time.localtime())
                            print(f'Time: {end_time} ')
                            y_test= np.array([1]*len(losses_seizure) + [0]*len(losses_normal))
                            y_pred=np.array([1 if l > THRESHOLD else 0 for l in losses_seizure+losses_normal])
                        
                            
                            clf_report = classification_report(y_test, y_pred, output_dict=True)
                            print(classification_report(y_test, y_pred))
                            plot_clasification_report(clf_report, MODEL_NAME)
                            
                            cf_matrix=confusion_matrix(y_test, y_pred)
                            plot_confusion_matrix(cf_matrix, MODEL_NAME)
                            
                            
                            if KFOLD_MODE:
                                print(f'\n######### KFOLD: {idx} #########   TRAIN: {len(train_index)} TEST: {len(test_index)}')
                            
                            print(f'Correct normal predictions: {correct_normal}/{len(test_normal)} -- {correct_normal/len(test_normal)*100}')
                            print(f'Correct seizure predictions: {correct_anomaly}/{len(test_seizure)} -- {correct_anomaly/len(test_seizure)*100}\n')
                            
                        else:
                            print('Error model action')
                            break
                        
                        if not KFOLD_MODE: break
                    
                    
                        #Correct normal predictions: 252/270 -- 93.33333333333333
                        #Correct anomaly predictions: 251/270 -- 92.96296296296296
                        
                    

                    
                    