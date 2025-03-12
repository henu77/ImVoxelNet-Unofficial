# 读取pkl文件
import pickle

# 读取pkl文件
with open(r'D:\Project\mmdetection3d\data\sunrgbd\sunrgbd_total_infos_val.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)