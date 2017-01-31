import pickle
from os import listdir
from os.path import isfile, join

import numpy as np
from scipy import misc


def func(img):
    print("hi")
    ltxt = open("data/label_colors.txt")
    ls = []
    for line in ltxt:
        ls.append([int(n) for n in line.split("\t")[0].split(" ")])

    labs = np.zeros((720, 960, 32), dtype=np.int32)
    for x in range(960):
        for y in range(720):
            truth = [np.array_equal(img[y, x], l) for l in ls]
            if any(truth):
                labs[y, x, truth.index(True)] = 1
    return labs


if __name__ == "__main__":
    mypath = "data/"
    labels = []
    images = []
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for name in onlyfiles:
        if name.endswith("L.png"):
            labels.append(misc.imread(mypath + name))
        elif name.endswith(".png"):
            images.append(misc.imread(mypath + name))
    images = np.array(images, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)


    from multiprocessing import Pool

    pool = Pool(4)
    results = pool.map(func, labels)

    with open('d.pickle', 'a') as f:
        pickle.dump(results, f)
