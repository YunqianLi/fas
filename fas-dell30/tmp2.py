import logging
import sys
import datetime


# 配置日志记录
logging.basicConfig(filename='./log/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 重定向标准输出到日志文件
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.rstrip():  # 防止空行被记录
            logging.log(self.level, message)

    def flush(self):
        pass


# 重定向标准输出和标准错误输出
sys.stdout = LoggerWriter(logging.INFO)
sys.stderr = LoggerWriter(logging.ERROR)

# 这些输出将被记录到日志文件中
print("This is a log message.")
print("Another log message.")
logging.error('ssss')
logging.info('sssssssss')


# mv -v ./train2/fake/FLU_PR* ./train2_bak/fake
# mv -v ./train2/fake/FLU_SG* ./train2_bak/fake
# mv -v ./val2/fake/FLU_PR* ./val2_bak/fake
# mv -v ./val2/fake/FLU_SG* ./val2_bak/fake
#
# cp -v ./train3_bak/fake/FLU_PR* ./train2/fake
# cp -v ./train3_bak/fake/FLU_SG* ./train2/fake
# cp -v ./val3_bak/fake/FLU_PR* ./val2/fake
# cp -v ./val3_bak/fake/FLU_SG* ./val2/fake
#
# rm ./train2/fake/FLU_PR*
# rm ./train2/fake/FLU_SG*
# rm ./val2/fake/FLU_PR*
# rm ./val2/fake/FLU_SG*

