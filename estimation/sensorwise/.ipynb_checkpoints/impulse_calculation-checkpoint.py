# 본 파일의 목적은 peak를 찾고 검출하는것임

# 필요 모듈 불러오기
from module.parse import *
from module.impulse import *
import multiprocessing


def process_work(model):
    p = Impulse(model)
    p.load()
    p.calImpulse_X()
    p.calImpulse_Y()
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
