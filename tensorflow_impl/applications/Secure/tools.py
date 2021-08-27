import random
import numpy as np

def divide_gradeint(gradient):
    rand = random.random()
    print("random numebr" , rand)
    return np.array(((gradient + rand) / 2 , (gradient - rand) / 2))


def compute_distances(grads):

    distances = np.zeros((len(grads) , len(grads)) , np.float)

    for i,a in enumerate(grads):
        for j,b in enumerate(grads):
            distances[i][j] = np.linalg.norm(a-b)

    return distances

def multi_krum_aggregator(distances, nbworkers, nbbyzwrks, number_of_gradient):
    nbselected = nbworkers - nbbyzwrks - 2
    scores = []
    for row in distances:
        scores.append(sum(np.sort(row)[0:nbselected]))
    
    idx = np.argpartition(scores, number_of_gradient)
    gar_weight = np.put(np.zeros((nbbyzwrks) , np.int), idx, 1)
    return gar_weight