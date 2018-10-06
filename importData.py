import pymysql
import traceback
import pandas as pd

"""
从数据库中查找数据
"""
def find_data():
    db = pymysql.connect(host="localhost",user="root",password="123456",db="stock",port=3306)
    cur = db.cursor()
    try:
        sql = "select * from finally_data"
        db.commit()
        cur.execute(sql)
        data_tuple = cur.fetchall()
        print("从数据中成功查询出数据")
        return data_tuple
    except Exception as e:
        print("查询数据库，出现错误")
        traceback.print_exc()
    finally:
        cur.close()
        db.close()
"""
将数据导出为csv格式
"""
def data_to_csv(file, data, columns):
    data = list(data)
    columns = list(columns)
    file_data = pd.DataFrame(data, index=range(len(data)), columns=columns)
    file_data.to_csv(file,index=False,encoding="GBK")
    print("导出成功")
