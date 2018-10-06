import pymysql
import traceback

#根据要求，整理数据，编辑成所需格式
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
        sql = "select * from history"
        cur.execute(sql)
        history_tuple = cur.fetchall()
        for history in history_tuple:
            for linked in linked_list:
                if history[1]==linked[1] and history[3]==linked[4]:
                    history_list = list(history)
                    history_list.append(linked[3])
                    finally_data.append(history_list)
        print(len(finally_data))
        for i in range(len(finally_data)):
            print(finally_data[i])
            sql = """update finally_data set morrow_opening_price=%f,morrow_top_price=%f,morrow_floor_price=%f,morrow_closing_price=%f,morrow_change_range=%f,
            morrow_turnover=%f,morrow_average_price=%f,morrow_turnover_rate=%f,morrow_amplitude=%f,morrow_is_harden='%s' 
            where stock_code='%s' and date='%s'""" % (finally_data[i][4],finally_data[i][5],finally_data[i][6],finally_data[i][7],finally_data[i][8],finally_data[i][9],
                                                      finally_data[i][10],finally_data[i][11],finally_data[i][12],finally_data[i][13],finally_data[i][1],finally_data[i][14])
            cur.execute(sql)
        #计算次日开盘涨幅
        sql = "UPDATE finally_data SET morrow_open_price_change = (morrow_opening_price-closing_price)/closing_price*100"
        cur.execute(sql)
        db.commit()
        print("成功提交任务")
    except Exception as e:
        print("出现错误，已回滚")
        db.rollback()
        traceback.print_exc()
    finally:
        cur.close()
        db.close()
#执行方法
search_data()