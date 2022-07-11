# 본 파일의 목적은 peak를 찾고 검출하는것임

# 실행코드
# python .\peak_detection.py -m woDgModel

# 필요 모듈 불러오기
from module.parse import *
from module.peak import *
import multiprocessing

# 해당하는 폴더 이름을 제공받고 해당 폴더 내에서 "moBWHT" 파일을 찾아서
#  peak와 peak timing 찾기
def process_work(model):
    p = Peak(model)
    p.load()
    p.findPeak_X()
    p.findPeak_Y()
    p.save()


def main():
    print("start")
    print("# of cpus", multiprocessing.cpu_count())
    models = ["DenseModel", "wDgMiniModel", "wDgModel", "woDgModel"]

    p = multiprocessing.Pool()

    p.map(process_work, models)
    print("Finish")


if __name__ == "__main__":
    main()
