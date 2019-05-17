import pandas as pd
import numpy as np
import time
import re # 正则表达式

path = "E:/tencent_data/"
save_path = path + "deal_data/"
total_exposurel_log = "totalExposureLog.out"# 历史曝光日志数据文件6G
user_data = "user_data" # 用户特征属性文件3G
ad_static_feature = "ad_static_feature.out" # 广告静态数据
ad_operation = "ad_operation.dat" # 广告操作数据
test_sample = "test_sample.dat" # 测试数据
test_sample_b = "Btest_sample_new.dat"
def timestamp_to_str(timestamp):
    """
    时间戳转字符串
    :param timestamp:
    :return:
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
def timestr_to_timestamp(timestr):
    """
    字符串转时间戳
    :return:
    """
    return int(time.mktime(time.strptime(timestr, "%Y-%m-%d %H:%M:%S")))
def deal_ad_static_feature():
    """
    01对广告静态数据进行处理
    :return:
    """
    df = pd.read_csv(path+ad_static_feature, header=None, low_memory=False, sep="\t")
    import_path = save_path + ad_static_feature.replace(".out", ".h5")
    old_row = df.shape[0]
    df = df.dropna(axis=0, how="any") # 去除有空值的行
    # 去除商品id为空的字段，即为-1，和出现多值的情况(有三条)
    df[3] = df[3].astype("str")
    df = df[~df[3].str.contains(",")]
    df[3] = df[3].astype("int64")
    df = df[~df[3].isin([-1])]
    # 去除广告行业id出现多值的情况
    df[5] = df[5].astype("str")
    df = df[~df[5].str.contains(",")]
    df[5] = df[5].astype("int64")
    # 去除创建时间异常的数据，异常基本为0
    df[1] = df[1].astype("int64")
    df = df[df[1]>100]
    new_row = df.shape[0]
    df.to_hdf(import_path, mode="w", key="df")
    print("广告静态数据处理完成，原数据%s条，新数据%s条，删除了%s条" % (old_row,new_row,old_row-new_row))

def deal_ad_operation():
    """
    02对广告操作数据进行处理
    :return:
    """
    df = pd.read_csv(path+ad_operation, header=None, low_memory=False, sep="\t")
    old_row = df.shape[0]
    # 去除广告静态数据里面没有的广告
    ad_static_feature_data = pd.read_hdf(save_path + ad_static_feature.replace(".out", ".h5"),key="df")
    valid_ad_id = ad_static_feature_data[0]
    df = df[df[0].isin(valid_ad_id)]
    # 去除错误的时间日期，如20190230000000,并转化为时间戳
    def timestr_to_timestamp(time_str):
        if time_str == "0":
            return 0
        try:
            time_array = time.strptime(time_str, "%Y%m%d%H%M%S")  # 字符串转时间数组（time.struct_time）
            timestamp = int(time.mktime(time_array))  # 字符串转化为时间戳
        except ValueError as e:
            return -1
        return timestamp
    df[1] = df[1].map(lambda x:timestr_to_timestamp(str(x)))
    df[1] = df[1].astype("int64")
    df = df[~df[1].isin([-1])]
    # 填充创建时间为0的数据（从广告静态数据中获取）
    def fill_zero(data):
        if data[1] == 0:
            ad_id = data[0]
            data[1] = ad_static_feature_data[1][ad_static_feature_data[0] == ad_id]
        return data
    df = df.apply(lambda row:fill_zero(row),axis=1)
    # 去除只有修改，没有新建的广告数据
    lt = {}
    del_id = []
    for index, row in df.iterrows():
        if row[0] in lt.keys():
            lt[row[0]].add(row[2])
        else:
            lt[row[0]] = {row[2]}
    for obj in lt.items():
        if 2 not in obj[1]:
            del_id.append(obj[0])
    df = df[~df[0].isin(del_id)]

    new_row = df.shape[0]
    df.to_hdf(save_path + ad_operation.replace(".dat", ".h5"), mode="w", key="df")
    print("广告操作数据处理完成，原数据%s条，新数据%s条，删除了%s条" % (old_row, new_row, old_row - new_row))

def get_intersection():
    """
    03求广告静态数据及广告操作数据共有的广告数据
    :return:
    """
    df_1 = pd.read_hdf(save_path + ad_operation.replace(".dat", ".h5"), key="df")
    df_2 = pd.read_hdf(save_path + ad_static_feature.replace(".out", ".h5"), key="df")
    intersection_id = set(df_1[0]).intersection(set(df_2[0]))
    df_1 = df_1[df_1[0].isin(intersection_id)]
    df_2 = df_2[df_2[0].isin(intersection_id)]
    df_1.to_hdf(save_path + ad_operation.replace(".dat", ".h5"), mode="w", key="df")
    df_2.to_hdf(save_path + ad_static_feature.replace(".out", ".h5"), mode="w", key="df")
    print("处理完成！")

def deal_total_exposurel_log():
    """
    04对历史曝光数据进行处理
    :return:
    """
    reader = pd.read_csv(path+total_exposurel_log, header=None, low_memory=False, sep="\t", chunksize=10000000)
    i = 0
    for chunk in reader:
        old_row = chunk.shape[0]
        chunk = chunk.dropna(axis=0, how="any")  # 去除有空值的行
        # 去除广告静态数据里面没有的数据
        ad_static_feature_data = pd.read_hdf(save_path + ad_static_feature.replace(".out", ".h5"), key="df")
        valid_ad_id = ad_static_feature_data[0]
        chunk = chunk[chunk[4].isin(valid_ad_id)]
        i += 1
        chunk.to_hdf(save_path + total_exposurel_log.replace(".out", "_%s.h5" % i), mode="w", key="df")
        new_row = chunk.shape[0]
        print("已导出文件%s，原数据%s条，新数据%s条，删除了%s条" % (i,old_row,new_row,old_row-new_row))
def merge_exposurel_log():
    """
    05合并曝光日志文件,并进行去重处理
    :return:
    """
    res = pd.DataFrame()
    for i in range(1,12):
        df = pd.read_hdf(save_path + total_exposurel_log.replace(".out", "_%s.h5" % i), key="df")
        res = res.append(df)

    res.drop_duplicates(keep="first", inplace=True)  # 去除完全重复的数据
    res.to_hdf(save_path + total_exposurel_log.replace(".out", ".h5"), mode="w", key="df")
    print("生成曝光日志文件，数据%s条" % res.shape[0])
def build_train_set():
    """
    06构建训练集
    :return:
    """
    op_df = pd.read_hdf(save_path + ad_operation.replace(".dat", ".h5"), key="df")
    st_df = pd.read_hdf(save_path + ad_static_feature.replace(".out", ".h5"), key="df")
    op_df.sort_values(by=[0,1], axis=0, ascending=True, inplace=True) # 对操作数据进行排序
    op_df.reset_index(drop=True, inplace=True) # 对索引进行重新排序

    columns = ["ad_id","create_time","material_size","industry_id","goods_type","good_id","ad_account_id",
               "target_time","target_people","bid","modify_time","ad_state","exposure_time"]
    train_set = pd.DataFrame(columns=columns)
    invaild_id = []  # 记录有误的ID，修改在创建之前
    for index,row in op_df.iterrows():
        ad_id = row[0]
        # 记录异常数据
        if (not train_set.empty) and (row[0] == train_set.iloc[-1]["ad_id"]):
            last_index = train_set.iloc[-1].name
            if row[2] == 2:
                if row[1] != train_set.iloc[-1]["modify_time"]:
                    if ad_id not in invaild_id:
                        print("创建时间唯一，不可能出现两个创建时间。")
                modify_field = row[3]
                if modify_field == 1:
                    train_set.loc[last_index,"ad_state"] = row[4]
                elif modify_field == 2:
                    if train_set.iloc[-1]["bid"] is not None:
                        raise Exception("新建的话，数据应该为空")
                    train_set.loc[last_index,"bid"] = row[4]
                elif modify_field == 3:
                    if train_set.iloc[-1]["target_people"] is not None:
                        raise Exception("新建的话，数据应该为空")
                    train_set.loc[last_index,"target_people"] = row[4]
                elif modify_field == 4:
                    if train_set.iloc[-1]["target_time"] is not None:
                        raise Exception("新建的话，数据应该为空")
                    train_set.loc[last_index,"target_time"] = row[4]

            else:
                # 判断修改日期时间是否为下两个自然日，不是的话，为新样本,广告状态为失效的话，也为旧样本
                last_modity_time = train_set.iloc[-1]["modify_time"]#now_time - now_time % (60*60*24) + time.timezone
                if train_set.loc[last_index,"ad_state"]==1 and (row[1] >= (last_modity_time+60*60*24*2-last_modity_time%(60*60*24)+time.timezone)):
                    train_set.append(train_set.iloc[-1],ignore_index=True)
                    last_index = last_index+1
                # 更新修改信息
                modify_field = row[3]
                if modify_field == 1:
                    train_set.loc[last_index,"ad_state"] = row[4]
                elif modify_field == 2:
                    train_set.loc[last_index,"bid"] = row[4]
                elif modify_field == 3:
                    train_set.loc[last_index,"target_people"] = row[4]
                elif modify_field == 4:
                    train_set.loc[last_index,"target_time"] = row[4]
                train_set.loc[last_index,"modify_time"] = row[1]
        else:
            modify_type = row[2]
            if modify_type != 2: # 广告一定是先创建再修改
                invaild_id.append(ad_id)
                print("业务逻辑错误：广告一定是先创建后修改",ad_id)
            create_time = row[1]
            ad_st_data = st_df.loc[st_df[st_df[0]==ad_id].index[-1]] # 广告对应的静态数据
            material_size = ad_st_data[6]
            industry_id = ad_st_data[5]
            goods_type = ad_st_data[3]
            goods_id = ad_st_data[4]
            ad_account_id = ad_st_data[2]
            target_time = None
            target_people = None
            bid = None
            modify_time = create_time
            ad_state = None  # 1正常，0失效
            exposure_time = create_time
            # 填充数据
            modify_field = row[3]
            if modify_field == 1:
                ad_state = row[4]
            elif modify_field == 2:
                bid = row[4]
            elif modify_field == 3:
                target_people = row[4]
            elif modify_field == 4:
                target_time = row[4]
            row = pd.Series([ad_id,create_time,material_size,industry_id,goods_type,goods_id,ad_account_id,target_time,
                             target_people,bid,modify_time,ad_state],index=columns)
            train_set = train_set.append(row, ignore_index=True)
    train_set.to_csv(save_path+"train_set.csv", sep="\t", header=None, index=None, encoding="utf-8")
    print("广告出现先修改后创建的情况，共有%s条" % len(invaild_id))

def generate_label():
    """
    根据构造好的训练集，生成标签
    :return:
    """
    df = pd.read_csv(save_path + "train_set.csv", sep="\t",header=None)
    df.loc[1,1] = timestamp_to_str(df.loc[1,1])
    df.loc[1, 10] = timestamp_to_str(df.loc[1, 10])
    # print(df[df[11]!=0]) #562,705,1175,1307
    log_df = pd.read_hdf(save_path + total_exposurel_log.replace(".out", ".h5"), key="df")
    log_df.sort_values(by=[1, 6], axis=0, ascending=True, inplace=True)  # 对操作数据进行排序
    result = log_df[log_df[4]==3230]
    result.loc[:,1] = result.loc[:,1].map(lambda x: timestamp_to_str(x))
    # log_df[1] = log_df[1].map(lambda x:timestamp_to_str(x))
    print(result[[1,6]])
    pass
def get_target_time_str(value):
    """
    根据腾讯投放时间的规则，转化为可视化
    :return:
    """
    result = []
    for vl in value.split(","):
        time_list = []
        for sre_match in re.finditer(r'[1]+','{:048b}'.format(int(vl))):
            index =sre_match.span()
            start = 48 - index[1]
            end = 48 - index[0]
            time_list.append("%02d:%02d-%02d:%02d" % (start//2,start%2,end//2,end%2))
        result.append(",".join(time_list))
    return ";".join(result)
def get_submission():
    """
    进行预测，生成比赛所需文件
    :return:
    """
    df = pd.read_csv(path+test_sample_b, header=None, low_memory=False, sep="\t")
    res = pd.DataFrame()
    for index,row in df.iterrows():
        res = res.append(pd.Series([row[0], row[10]]), ignore_index=True)
    res[0] = res[0].astype("int64")
    res.to_csv(save_path+"submission.csv", sep=",", header=None, index=None, encoding="utf-8")
    print("已生成提交文件")
# df = pd.read_csv(save_path + ad_operation.replace(".dat", ".csv"), sep="\t",header=None)
# index = df[df[3]==4].index.tolist()
# df.iloc[index,4] = df.iloc[index,4].map(lambda x:get_target_time_str(x))
# deal_total_exposurel_log()
# get_intersection()
# df = pd.read_csv(path+test_sample, header=None, low_memory=False, sep="\t")
# merge_exposurel_log()
# df = pd.read_hdf(save_path + total_exposurel_log.replace(".out", ".h5"), key="df")
# get_submission()
# build_train_set()
# generate_label()
# df = pd.read_csv(save_path + "train_set.csv", sep="\t",header=None)


