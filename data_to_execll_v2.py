import os
import glob
import pandas as pd



coloum_headings=[]
total_rows=[]
folder_modified=''
def contents_rows_check(text_file,name,folder):
    data=pd.DataFrame()
    for line in text_file:
        content_separator = line.split('\t')
        coloum_headings.append(content_separator)
        total_rows.append(len(content_separator))

    df = pd.DataFrame(coloum_headings[1:])
    folder_modified="Excell_data_limited_2/"+folder
    if not os.path.exists(folder_modified):
        os.system(f"mkdir -p {folder_modified}")
    excel_file_name = folder_modified+'/'+name+'.xlsx'
    df.to_excel(excel_file_name, index=False)
    print(f'Data saved to {excel_file_name}')
    return(len(total_rows))

def contents_rows_check_limited(text_file,name,folder):
    data=pd.DataFrame()
    for line in text_file:
        content_separator = line.split('\t')
        coloum_headings.append(content_separator)
        total_rows.append(len(content_separator))
    df = pd.DataFrame(coloum_headings[1:])
    df = df[desired_feature_sets]
    folder_modified="Excell_data_limited/"+folder
    if not os.path.exists(folder_modified):
        os.system(f"mkdir -p {folder_modified}")
    excel_file_name = folder_modified+'/'+name+'.xlsx'
    df.to_excel(excel_file_name, index=False)
    print(f'Data saved to {excel_file_name}')
    return(len(total_rows))
def text_folder(file,count,name):
    text_file=open(file, "r")
    for lines in text_file:
        if ("Test point data:" in lines):
            rows=contents_rows_check(text_file,name,str(file.split('.')[0].split("/")[-2].replace(' ','_')))
            print(f"Total_rows:{rows},\non data lines:{count}\nprocess_completes")
            break
        else:
            count+=1
def main():
    count = 0
    data_path=input("please give the data path")
    folder_content=os.listdir(data_path)
    txt_files = glob.glob(os.path.join(data_path, '**/*.txt'), recursive=True)
    for paths in range(0,len(txt_files)):
        text_folder(txt_files[paths],count,str(txt_files[paths].split('.')[0].split("/")[-1]))
        coloum_headings.clear()
        total_rows.clear()
        count = 0

if __name__ == "__main__":
    main()

