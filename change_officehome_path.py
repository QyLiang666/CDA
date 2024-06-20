import os

# 打开原始文件和临时文件
with open('data/office-home/Clipart.txt', 'r') as f, open('data/office-home/Clipart_temp.txt', 'w') as f_temp:
    # 逐行读取原始文件
    for line in f:
        # 去除每行中的 /images 字段
        line_new = line.strip().replace('/images', '')
        # 将处理后的内容写入临时文件
        f_temp.write(line_new + '\n')

# 关闭原始文件和临时文件
f.close()
f_temp.close()

# 删除原始文件，并将临时文件重命名为原始文件的名称
os.remove('data/office-home/Clipart.txt')
os.rename('data/office-home/Clipart_temp.txt', 'data/office-home/Clipart.txt')