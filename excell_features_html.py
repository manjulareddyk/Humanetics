import pandas as pd
import glob
import os

def replace(word):
    word_list=[' ','/','(',')','[',']','-','°','%']
    for words in word_list:
        word=word.replace(words,'_')
    return word

def preprocess_excell_data_types(df):
    df = df.head(2000)
    df = df.convert_dtypes()
    skip_coloumns = ['\n', 'Source', 'Safety flags #1 [-]', 'Safety flags #2 [-]', 'Safety flags #3 [-]',
                    'Safety flags #4 [-]', 'Relay State [-]', 'Time (UTC)', 'Index [-]',
                    'Int. desired speed [kph]', 'Int. desired steering radius [m]', 'Int. desired steering angle [°]',
                    'Desired steering speed [-]', 'Steering position [-]', 'Desired brake pressure [Bar]',
                    'Brake pressure [Bar]', 'Desired brake position [-]', 'Brake position [-]',
                    'Desired brake speed [-]',
                    'Way from motors [m]', 'Motor [L] Brake [-]', 'Motor [L] State [-]', 'Motor [R] Brake [-]',
                    'Motor [R] State [-]', 'Motor [FL] Brake [-]', 'Motor [FL] State [-]', 'Motor [FR] Brake [-]',
                    'Motor [FR] State [-]', 'Slip front left [%]', 'Slip front right [%]', 'RPM front left [-]',
                    'RPM front right [-]', 'Time to meeting pos [s]', 'VUT distance to crashpoint [m]',
                    'VUT time to meeting pos [s]', 'VUT long distance [m]', 'VUT lat distance [m]', 'VUT speed [kph]',
                    'VUT Pos E [m]', 'VUT Pos N [m]', 'VUT Pos X [m]', 'VUT Pos Y [m]', 'VUT distance to UFO [m]',
                    'VUT side deviation [m]', 'TTC long [s]', 'TTC lat [s]', 'TTC abs [s]',
                    'Network roundtrip time [ms]',
                    'Additional side offset [m]', 'Additional forward offset [m]', 'Additional offset blend factor [1]',
                    'Main buffer batt voltage [V]', 'GNSS buffer batt voltage [V]', 'Debug F1 [-]', 'Debug F2 [-]',
                    'Debug F3 [-]', 'Debug I1 [-]', 'Debug I2 [-]', 'Debug I3 [-]']
    skdf = df.drop(columns=skip_coloumns)
    # Replace commas with periods and convert to float
    for columns in skdf.columns:
        skdf[columns] = skdf[columns].astype(str).str.replace(',', '.', regex=True).astype(float)

    df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'], format='%d/%m/%Y %H:%M:%S,%f')
    skdf['Time (UTC)'] = df['Time (UTC)']
    return skdf


def labeling(ldf,path):
    label_map = {
        0: ['ok', 'belt-0', 'belt-50', 'ok-first-day'],
        1: ['flat-unknown-rul']
    }
    for key,values in label_map.items():
        if any (value in path for value in values):
            ldf['class_labels']=key
    return ldf

def combinedfeatures(full_files):

    df = pd.read_excel(full_files[0], header=1)
    df = df.head(2000)
    df = preprocess_excell_data_types(df)
    df=labeling(df,full_files[0])
    list_words=df.columns.to_list()

    word_list = []
    for words in list_words:
        words = replace(words)
        word_list.append(words)
    map_dict={}
    for id, creating_dic in enumerate(list_words):
        map_dict[f'{creating_dic}']=word_list[id]
    for files in full_files[1:]:
        df2 = pd.read_excel(files, header=1)
        df2 = df2.head(2000)
        df2 = preprocess_excell_data_types(df2)
        df2= labeling(df2, files)
        df = pd.concat([df, df2],ignore_index=True)
        print(f"merging file: {files}")
    folder_name = "combinedfeatures/"
    if not os.path.exists(folder_name):
        os.system(f"mkdir -p {folder_name}")
    df.rename(columns=map_dict,inplace=True)
    df.to_excel(f"{folder_name}combinedfeatures.xlsx")
    df.to_csv(f"{folder_name}combinedfeatures_utf-8.csv",encoding='utf-8',index_label='Index')
    df.to_csv(f"{folder_name}combinedfeatures.csv", index_label='Index')




def main():

    path = input("please give data the excell path")
    excell_file = glob.glob(os.path.join(path, '**/*.xlsx'), recursive=True)
    combinedfeatures( excell_file)


if __name__=="__main__":
    main()

