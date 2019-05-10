import os
import shutil
def readInChunks(fileObj, chunkSize=1024*1024*100):
    while 1:
        data = fileObj.readlines(chunkSize)
        if not data:
            break
        yield "".join(data)
def cut_file(path):
    f = open(path)
    i=0
    export_path = "./data/totalExposureLog/"
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    os.mkdir(export_path)
    for chuck in readInChunks(f):
        i=i+1
        wrfile=open(export_path+"totalExposureLog_"+str(i)+".out",'w')
        wrfile.write(chuck)
        print("已生成文件：totalExposureLog_"+str(i)+".out")
        wrfile.close()
    f.close()
file_path= './data/totalExposureLog/totalExposureLog_1.out'
cut_file(file_path)