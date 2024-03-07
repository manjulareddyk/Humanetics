import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#from sklearn.model_selection import train_test_split

samples=100
batch_size = 512
df2=pd.DataFrame()

def json_file_write(file_path, my_dict):
    # Convert NumPy arrays to Python lists
    for key, value in my_dict.items():
        if isinstance(value, np.ndarray):
            my_dict[key] = value.tolist()

    # Write dictionary to JSON file
    with open(file_path, 'w') as json_file:
        json.dump(my_dict, json_file, indent=4)  # indent parameter for pretty formatting

class Net_test(nn.Module):
    def __init__(self, ):
        super(Net_test, self).__init__()
        self.fc1 = nn.Linear(samples, samples * 3)
        self.fc2 = nn.Linear(samples * 3, 400)
        self.fc3 = nn.Linear(400, 100)
        self.fc4 = nn.Linear(100, 2)
        self.bn1 = nn.BatchNorm1d(samples * 3)  # BatchNorm1d after the first linear layer
        self.bn2 = nn.BatchNorm1d(400)  # BatchNorm1d after the second linear layer
        self.bn3 = nn.BatchNorm1d(100)  # BatchNorm1d after the third linear layer

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.softmax(self.fc4(x), dim=1)
        return x
def step_average(sample_array_a,start,step):
    back_sum=sample_array_a[(start-step):start].sum()
    front_sum=sample_array_a[start:(start+step)].sum()
    avg=((back_sum+front_sum)/(step*2))
    return avg

def average_replace_values_v2(sample_array_d,start_index1,start_index2,step):
    range_step=start_index2-start_index1
    final_array=sample_array_d.copy()
    for values in range(0,range_step+1):
        avg=step_average(final_array,(values+start_index1),step)
        sample_array_d[values+start_index1]=avg
    return sample_array_d


def label_index_fun(orginal, label_list):
    index_list = dict()
    for label in label_list:
        labels_check = label_0 = orginal[:, -1] == label
        index_list[label] = np.where(labels_check)

    print(index_list)
    return index_list
def spliting_array(orginal, split_ratio_train=80, split_ratio_test=20, validation_split_ratio=0, labels_list=None):
    label_length = []
    label_index = label_index_fun(orginal, labels_list)
    for keys in label_index.keys():
        label_length.append(len(label_index[keys][0]))
    print(f"orginal_array shape:{orginal.shape}")
    train_array = np.zeros(shape=((((orginal.shape[0]) * split_ratio_train) // 100), orginal.shape[1]), dtype=float)
    test_array = np.zeros(shape=((((orginal.shape[0]) * split_ratio_test) // 100) + 1, orginal.shape[1]), dtype=float)
    validation_array = np.zeros(shape=((((orginal.shape[0]) * validation_split_ratio) // 100) + 1, orginal.shape[1]), dtype=float)
    print(f"train shape:{train_array.shape}")
    print(f"test shape:{test_array.shape}")
    print(f"validation shape:{validation_array.shape}")
    start_train = 0
    start_test = 0
    start_validation=0
    for label_class, values in enumerate(label_length):
        train_ratio = (((values * split_ratio_train) // 100))
        test_ratio = ((values * split_ratio_test) // 100)
        validation_ratio = ((values * validation_split_ratio) // 100)
        train_array[start_train:(train_ratio + start_train)] = orginal[label_index[label_class][0][0:(train_ratio)]]
        test_array[start_test:(test_ratio + start_test)] = orginal[label_index[label_class][0][(train_ratio):(train_ratio + test_ratio)]]
        validation_array[start_validation:(validation_ratio + start_validation)] = orginal[label_index[label_class][0][(train_ratio + test_ratio):(train_ratio + test_ratio+validation_ratio)]]
        start_train = train_ratio
        start_test = test_ratio
        start_validation = validation_ratio
    return train_array, test_array,validation_array

def pytorch_training(train_data,test_data,Model_Name,samples):
    #step=train_data.shape[0]
    #samples=20
    class Net(nn.Module):
        def __init__(self,):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(samples, samples * 3)
            self.fc2 = nn.Linear(samples * 3, 400)
            self.fc3 = nn.Linear(400, 100)
            self.fc4 = nn.Linear(100, 2)
            self.bn1 = nn.BatchNorm1d(samples * 3)  # BatchNorm1d after the first linear layer
            self.bn2 = nn.BatchNorm1d(400)  # BatchNorm1d after the second linear layer
            self.bn3 = nn.BatchNorm1d(100)  # BatchNorm1d after the third linear layer

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            x = F.softmax(self.fc4(x), dim=1)
            return x
    train_tensor = torch.from_numpy(train_data.astype('float32'))
    test_tensor = torch.from_numpy(test_data.astype('float32'))
    #num_features = train_data.shape[0]
    # Define the neural network
    net = Net()
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move the model to GPU
    net.to(device)

    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


    epochs = 100
    # Training loop
   
    batch_size = 512  # Define your batch size
    training_loss_dict=dict()

    for epoch in range(epochs):
        # Iterate over the entire dataset in batches
        # Get the current time before the operation
        start_time = time.time()
        for batch_start in range(0, len(train_tensor), batch_size):
            batch_end = min(batch_start + batch_size, len(train_tensor))
            batch_train = train_tensor[batch_start:batch_end].to(device)  # Move data to GPU

            inputs_train = batch_train[:, :-1]
            labels_train = batch_train[:, -1].long()

            optimizer.zero_grad()
            outputs_train = net(inputs_train)
            loss_train = criterion(outputs_train, labels_train)
            loss_train.backward()
            optimizer.step()

        # Get the current time after the operation
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"loss_train:{loss_train}====epoch:{epoch}====Elapsed time: {elapsed_time} seconds ")
        training_loss_dict[epoch] = loss_train.cpu().detach().numpy()# Convert loss_train to CPU before storing in dictionary
    # Plot the loss graph after all epochs
    plt.plot(training_loss_dict.keys(), training_loss_dict.values())
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(Model_Name)

    # Create directory to save plots if it doesn't exist
    path=os.getcwd()
    path=f'{path}/loss_graphs/{Model_Name}'
    os.makedirs(path, exist_ok=True)

    # Save the plot
    fig_path = f"{path}/{Model_Name}_{batch_size}.png"
    plt.savefig(fig_path)
    plt.close()
    # Show the plot
    #plt.show()
    training_loss_dict['Epocs'] = epoch
    training_loss_dict['Btach'] = batch_size
    path=os.getcwd()
    path=f'{path}/Trained_Model/{Model_Name}'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # Save the trained model
    model_path = f"{path}/{Model_Name}_{batch_size}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_train,
    }, model_path)
    json_path = f"{path}/{Model_Name}_{batch_size}.json"
    #json_file_write(json_path,training_loss_dict)
    del net

    # Testing loop
    # Load the saved model
    loaded_model = Net()
    loaded_model.to(device)
    loaded_checkpoint = torch.load(model_path)
    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    loaded_model.eval()

    # Testing loop
    correct = 0
    total = 0

    with torch.no_grad():
        # for batch_test in test_tensor[:, :data_stop, :]:
        for batch_test in test_tensor.unsqueeze(0):
            inputs_test = batch_test[:, :-1].to(device)
            labels_test = batch_test[:, -1].long().to(device)
            #print(f"label:{labels_test}=={batch_test}")
            outputs_test = loaded_model(inputs_test) # Use the loaded model for testing
            _, predicted = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()

    accuracy = correct / total
    print(f'Testing Accuracy {Model_Name}: {accuracy * 100:.2f}%')
    training_loss_dict['Testing Accuracy'] = accuracy
    training_loss_dict['Epocs'] = epoch
    training_loss_dict['Btach'] = batch_size
    json_file_write(json_path, training_loss_dict)
    #del net
    del loaded_model

def load_pytorch_model(model_path,test_tensor,Model_Name,concatenated_array,index_c):
    # Load the saved model
    loaded_model = Net_test()
    loaded_checkpoint = torch.load(model_path)
    loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    loaded_model.eval()

    # Testing loop
    correct = 0
    total = 0
    print(f"index_g:{index_c}")
    test_tensor = torch.from_numpy(test_tensor.astype('float32'))
    with torch.no_grad():
        # for batch_test in test_tensor[:, :data_stop, :]:
        for batch_test in test_tensor.unsqueeze(0):
            inputs_test = batch_test[:, :-1]
            labels_test = batch_test[:, -1].long()
            #print(f"label:{labels_test}=={batch_test}")
            outputs_test = loaded_model(inputs_test)  # Use the loaded model for testing
            #print(f"outputs_test:{outputs_test}")
    concatenated_array[:test_tensor.shape[0],index_c:index_c+2]=outputs_test
    return concatenated_array

def training_samples(original_array,featurtes_list,values,concatenated_array,index_c):
    print(f"Received array shape:{original_array.shape}")
    back_sample_length=100
    samples_location_start=100
    features_length=original_array.shape[0]
    features_sample_length=original_array.shape[1]
    sample_data=original_array.shape[2]
    sub_sample_index=0
    sub_samples = features_sample_length - samples_location_start + 1
    trainig_array=np.zeros(shape=(1,(features_sample_length*(sample_data-back_sample_length)+1),(back_sample_length+1)),dtype=float)
    features=0
    sub_sample_index = 0
    for samples_features in range(0,(features_sample_length)):
        for sample in range(samples_location_start,sample_data):
            trainig_array[features][sub_sample_index][0:back_sample_length]=original_array[values][samples_features][(sample-back_sample_length)+1:sample+1]
            if (sample<2000):
                trainig_array[features][sub_sample_index][-1]=0
            elif (sample>2000):
                trainig_array[features][sub_sample_index][-1] =1
            sub_sample_index +=1
    train_data_u, test_data_u ,validate_array_u= spliting_array(trainig_array[features], labels_list=[0, 1], split_ratio_test=20,split_ratio_train=70,validation_split_ratio=10)
    path=os.getcwd()
    path=f'{path}/Trained_Model/'
    print(f"model path :{path}{featurtes_list[values]}/{featurtes_list[values]}.pth")
    concatenated_array2=load_pytorch_model(f"{path}{featurtes_list[values]}/{featurtes_list[values]}_{batch_size}.pth",train_data_u, featurtes_list[values],concatenated_array,index_c)
    df2.to_excel('Trained_Model_Accuracy.xlsx')
    return concatenated_array2,train_data_u

def main():
    file_csv=input('please give the csv file')
    df=pd.read_csv(file_csv)
    df2=df
    skip_coloums=['Time__UTC_']
    df2=df2.drop(columns=skip_coloums)
    featurtes_list = df2.columns.to_list()
    array_df = np.zeros(shape=(len(df2.columns.to_list()), 360, 4000), dtype=float)
    print(f"array shape:{array_df.shape}")

    index_c=0
    class_index_1 = df2[df2['class_labels'] == 1].index[0]
    print(class_index_1)
    class_1_samples = 10
    sample_files = 36
    total_sample = sample_files * class_1_samples
    step_index = class_index_1
    step=100
    #featurtes_list=['Batt_State_of_charge_2__%_','Slip_rear_right__%_','GPS_Num_satellites____','Batt_Current_1__A_','Acc_Z__m_ss_','Motor__R__RPM__1_min_','AngRate_Z____s_','Slip_rear_left__%_','Change_of_Heading____s_','Acc_Y__m_ss_']
    featurtes_list=['Batt_State_of_charge_2____','Slip_rear_right____',]
    concatenated_array = np.zeros(shape=(982800, (len(featurtes_list) * 2) + 1))
    for index_f, features in enumerate(featurtes_list):
        start_sample = 0
        step_sample = 2000
        for samples in range(0, total_sample, class_1_samples):
            sample_array = np.zeros(4000)
            for index_1_sample in range(0, class_1_samples):
                sample_array[0:2000] = df2[features][start_sample:start_sample + 2000]
                sample_array[2000:] = df2[features][step_index:step_index + 2000]
                avg=average_replace_values_v2(sample_array, 1899, 2299,step)#average step
                array_df[index_f][samples + index_1_sample] = avg
                step_index = step_index + 2000
            start_sample = start_sample + 2000
            step_index = class_index_1
        train_array,label_array=training_samples(array_df,featurtes_list,index_f,concatenated_array,index_c)
        index_c=index_c+2
    train_array[:, -1] = label_array[:, -1]
    train_data_u, test_data_u ,validate_array_u= spliting_array(train_array, labels_list=[0, 1], split_ratio_test=20,split_ratio_train=70,validation_split_ratio=10)
    #np.random.shuffle(train_data_u)
    pytorch_training(train_data_u,test_data_u,"model",((len(featurtes_list) * 2) ))



if __name__=="__main__":
    main()