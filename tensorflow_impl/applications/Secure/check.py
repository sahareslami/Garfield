import pickle

with open("worker9090.pickle", 'rb') as handle:
    worker_1_grads = pickle.load(handle)


with open("worker9093.pickle", 'rb') as handle:
    worker_2_grads = pickle.load(handle)

with open("worker_server_gradients.pickle", 'rb') as handle:
    worker_server_gradients = pickle.load(handle)

with open("model_server_gradients.pickle", 'rb') as handle:
    model_server_gradients = pickle.load(handle)

print("worker one gradients")
print(worker_1_grads)
print("----------------------------------------")
print("worker two gradients")
print(worker_2_grads)
print("----------------------------------------")

list_of_grads = []

for grads1 , grads2 in zip(worker_server_gradients , model_server_gradients):
    print("worker server gradients")
    print(grads1)
    print("-------------------------")
    print("model server gradients")
    print(grads2)
    print("-------------------------")

    
average = (worker_2_grads + worker_1_grads) / 2

with open("worker_server_gradient_in_model_Server.pickle", 'rb') as handle:
    worker_server_gradient_in_model_Server = pickle.load(handle)

print("worker_server_gradient_in_model_Server" , worker_server_gradient_in_model_Server)
print(worker_server_gradients[0] + worker_server_gradients[1])
with open("final_grads.pickle", 'rb') as handle:
    final_grads = pickle.load(handle)
  
print(average , len(average[0]))
print(final_grads , len(final_grads))
