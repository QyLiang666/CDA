import os
#本代码用于将作者提供的txt文件中的路径改为本机的绝对路径，即 通过字符串拼接，将图像文件映射到当前主机绝对路径下的某个文件
#folder是作者提供的、处理好的各数据集的根路径（相对路径），建议将folder路径所指的文件和当前.py文件在同一目录下，例如：可以都放在项目的根目录下
#本代码要求数据集必须存储在绝对路径/home/liangqiyue/datasets/VISDA;/home/liangqiyue/datasets/OfficeHome/OfficeHome;/home/liangqiyue/datasets/DomainNet路径下，差一点都不行！
#第29行中的删除每行的前几个片段，需要根据txt的实际路径决定，不同项目的作者的前缀路径写的不一样。 数一下想要那个字段在第几个即可，从0开始，但因为每行都以/开头，所以以/为分界线截断后得到的列表中第一个元素一定是“空”！
folder = 'data'                     
datasets = os.listdir(folder)
for dataset in datasets:        # 逐一处理folder目录下的每个数据集

    # 这里改名并不是改folder目录下的文件名，而是用于下面拼接字符串的
    if dataset[:3].lower() == "vis":
        dataset_name = "VISDA/"     
    if dataset[:3].lower() == "off":
        dataset_name = "OfficeHome/OfficeHome/"
    if dataset[:3].lower() == "dom":
        dataset_name = "DomainNet/"

    txt_files_path = []        # 用来装每个数据集目录下的全部txt文件
    dataset_path = os.path.join(folder, dataset)
    for file in os.listdir(dataset_path): # 遍历目录下的所有文件和目录
        file_path = os.path.join(dataset_path, file)  # 构造文件的完整路径
        if os.path.isfile(file_path) and file.endswith(".txt"):
            txt_files_path.append(file_path)  # 如果是文件且以 .txt 结尾，将其路径添加到列表中
    # print(txt_files_path)

    for txt_file_path in txt_files_path:
        # 打开文件
        with open(txt_file_path, 'r+') as f:
            # 逐行读取文件内容
            lines = f.readlines()
            # 将每一行按照"/"分割,删除0-5片段,这里是从第六个片段开始，并在每行的最开始补充绝对路径的开头（因数据集而异！）
            new_lines = ["/home/liangqiyue/datasets/"+dataset_name+"/".join(line.strip().split("/")[6:])+"\n" for line in lines]
            # 将修改后的内容写回原文件
            f.seek(0)
            print(new_lines)
            f.writelines(new_lines)  
            f.truncate()    
