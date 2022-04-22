from multiprocessing import Pool
from multiprocessing import Process
from os import getpid
import time


def double(i,j):
    print(getpid())
    print(i+j)

if __name__ == '__main__':
    start = time.time()
    for i in range(100):
        p=Process(target = double,args = (10,15))
        p.start()
    p.join()
    print(time.time()-start)
    # for i in range(100):
    #     double(10,15)
    # print(time.time()-start)