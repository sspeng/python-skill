import pymysql
import  traceback


def search_data():
    finally_data = []
    linked_list = []
    db = pymysql.connect(host="localhost", user="root", password="123456", db="stock", port=3306)
    cur = db.cursor()
    try:
        sql = "select * from linked"
        cur.execute(sql)
        linked_tuple = cur.fetchall()
        #构造关联的list
        for each_linked_tuple in linked_tuple:
            linked_list.append(list(each_linked_tuple))
        sql = "select * from history where pk_stock = 3589"
        cur.execute(sql)
        history_tuple = cur.fetchall()
        for history in history_tuple:
            for linked in linked_list:
                if history[1]==linked[1] and history[3]==linked[4]:
                    finally_data.append(list(history))
                    print("获取数据：",list(history))
        print(finally_data)

        # print(history_tuple[0])
        # print(len(history_tuple))
    except Exception as e:
        print("出现错误，已回滚")
        db.rollback()
        traceback.print_exc()
    finally:
        cur.close()
        db.close()
search_data()