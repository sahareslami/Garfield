# coding: utf-8
###
 # @file   trainer.py
 # @author  Anton Ragot <anton.ragot@epfl.ch>, Jérémy Plassmann <jeremy.plassmann@epfl.ch>
 #
 # @section LICENSE
 #
 # MIT License
 #
 # Copyright (c) 2020 Distributed Computing Laboratory, EPFL
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in all
 # copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
###

#!/usr/bin/env python

import argparse
import pickle
import time
import os
import numpy as np
import sys
import pickle


from libs.worker import Worker
from libs.ps import PS
from libs.byz_worker import ByzWorker
from libs import tools as gadget

from libs.secure_worker import SecureWorker
from libs.model_server import ModelServer
from libs.worker_server import WorkerServer
from rsrcs.aggregator_tf.aggregator import Aggregator_tf

from rsrcs.network import Network
import tools


# Allowing visualization of the log while the process is running over ssh
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


FLAGS = None

def main():
    n = Network(FLAGS.config)

    if n.get_task_type() == 'worker':
        # print("I am worker, and I am here")
        if n.get_my_attack() != 'None':
            w = ByzWorker(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        else:
            w = SecureWorker(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        w.start()
        model_aggregator = Aggregator_tf(n.get_model_strategy(), len(n.get_all_workers()), FLAGS.nbbyzwrks, FLAGS.native)

        for iter in range(FLAGS.max_iter+1):
            # print("I am worker and starting the worker" , iter)
            models = w.get_models(iter)
            # print("I am worker and I got the model successfully" , iter)
            aggregated_model = model_aggregator.aggregate(models)
            w.write_model(aggregated_model)
            # print("I am worker and I write the model" , iter)
            loss, grads = w.compute_gradients(iter)
            # if(iter == 0):
                # with open('worker' + w.port + '.pickle', 'wb') as handle:
                        # pickle.dump(grads , handle)
            # print("I am worker and I compute the gradient, yay ", iter, "\n",  grads)
            # print("the type" , type(grads))
            # print("the shape" , grads[0] , len(grads[0]))
            # print("the the len" , len(grads))
            partial_grads = tools.divide_gradeint(grads)
            # print("divided grads shape is" , partial_grads)
            # print("each one len", len(partial_grads[0]) , len(partial_grads[1]))
            # print("I am worker and I divide the gradient successfully", iter , partial_grads)
            # print("first one:" , iter , "\n" , partial_grads[0] , len(partial_grads[0]) , partial_grads[0].shape)
            # print("second one:" , iter, "\n" ,  partial_grads[1], len(partial_grads[1]) , partial_grads[1].shape)
            w.commit_gradients(partial_grads)
            # print("I am worker and commit done succefully:" , iter)
        w.stop(1)


    elif n.get_task_type() == 'ws':
        # print("I am worker server, and I am here")
        p = WorkerServer(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        p.start()

        gradient_aggregator = Aggregator_tf(n.get_gradient_strategy(), len(n.get_all_workers()), FLAGS.nbbyzwrks, FLAGS.native)

        accuracy = 0
        for iter in range(FLAGS.max_iter):
            # print("worker server, I am here" , iter)
            models = p.get_models(iter)
            # print("I am worker server, and I get the model successfully" , iter)
            p.write_model(models[0])
            # print("I am worker server, and the model write succefully" , iter)
            partial_gradients = p.get_gradients(iter)

            # if(iter == 1):
                # with open('worker_server_gradients.pickle', 'wb') as handle:
                        # pickle.dump(partial_gradients , handle)

            # print("I am worker server , and I got the partial gradients" , iter)
            # print("this is gradients that worker server received " , partial_gradients)
            # print("this is gradients that worker server received" , partial_gradients[0].shape , len(partial_gradients))
            partial_pairwise_distances = tools.compute_distances(partial_gradients)
            # print("I am worker server, and partial pairwise disntance compute successfully" , partial_pairwise_distances, type(partial_pairwise_distances))
            final_pairwise_distances = p.compute_final_pairwise_distances(partial_pairwise_distances, iter)
            # print("I am worker server, and final pairwise disntance compute successfully", iter , "\n" , final_pairwise_distances)
            aggregated_gradient_weight = tools.multi_krum_aggregator(final_pairwise_distances , 2 , 0 , 2)
            # print("I am worker server, and aggregate the " ,iter) 
            # print("I am worker server and this is the aggregate gradient weight" , aggregated_gradient_weight , type(aggregated_gradient_weight))
            # print("I am worker server and this is the partial gradient for worker server" , partial_gradients , type(partial_gradients))
            

            # print("I am worker server and this is what should be commited" , np.array((partial_gradients , aggregated_gradient_weight)))

            worker_server_final_gradeint = tools.compute_weighted_gradient(partial_gradients , aggregated_gradient_weight)
            p.commit_semi_gradient(np.array((worker_server_final_gradeint , aggregated_gradient_weight)))
            # print("I am worker server, and I commit the final partial gradient successfully" , iter)
            if iter == FLAGS.max_iter:
                p.commit_model(iter)
            

    elif n.get_task_type() == 'ms':
        # print("I am model server, and I am here")
        p = ModelServer(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        p.start()

        gradient_aggregator = Aggregator_tf(n.get_gradient_strategy(), len(n.get_all_workers()), FLAGS.nbbyzwrks, FLAGS.native)

        accuracy = 0
        for iter in range(FLAGS.max_iter):
            # print("I am model server, I am here:" , iter)
            models = p.get_models(iter)
            p.write_model(models[0])
            # print("I am model server and before getting gradient:" , iter)
            partial_gradients = p.get_gradients(iter)
            # if(iter == 0):
                # with open('model_server_gradients.pickle', 'wb') as handle:
                        # pickle.dump(partial_gradients , handle)
            # print("I am model server and it's after getting gradeint:", iter , partial_gradients)
            partial_pairwise_distances = tools.compute_distances(partial_gradients)
            # print("I am model server and it's my distances" , partial_pairwise_distances)
            # print("I am model server and this is distances:" , partial_pairwise_distances , type(partial_pairwise_distances) , len(partial_pairwise_distances))
            p.commit_partial_difference(partial_pairwise_distances)
            # print("I am model server and after commiting partial difference:" , iter)
            final_gradient = p.compute_final_gradient(iter , partial_gradients)
            # with open("model_server_gradients.pickle", 'rb') as handle:
                # model_server_gradients = pickle.load(handle)

            # print("I am model server and I got the final gradient:" , iter , final_gradient , type(final_gradient) , final_gradient.shape)
            # if(iter == 1):
                # with open('final_grads.pickle', 'wb') as handle:
                        # pickle.dump(final_gradient , handle)
            model = p.upate_model(final_gradient)

            # print("I am model server, and I update the model perfectly fine")
            p.commit_model(model)
            # print("I am model server, and I commit the model succefully")
            
            # gadget.training_progression(FLAGS.max_iter, iter, accuracy)
            if iter%50 == 0:
                accuracy = p.compute_accuracy()
 
        print("\nTraining done!")
        p.stop(1)

    else:
        print("Unknown task type, please check TF_CONFIG file")
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Flags for defining current Node
    parser.add_argument('--config',
                        type=str,
                        default="TF_CONFIG",
                        help='Config file location.')
    parser.add_argument('--log',
                        type=bool,
                        default=False,
                        help='Add flag to print intermediary steps.')
    parser.add_argument('--max_iter',
                        type=int,
                        default="2000",
                        help='Maximum number of epoch')
    parser.add_argument('--dataset',
                        type=str,
                        default="mnist",
                        help='Choose the dataset to use')
    parser.add_argument('--model',
                        type=str,
                        default="Small",
                        help='Choose the model to use')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Set the batch size')
    parser.add_argument('--nbbyzwrks',
                        type=int,
                        default=0,
                        help='Set the number of byzantine workers (necessary for Krum aggregation)')
    parser.add_argument('--native',
                        type=bool,
                        default=False,
                        help='Choose to use the native aggregators.')

    FLAGS, unparsed = parser.parse_known_args()
    main()
