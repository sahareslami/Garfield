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
import sys


from libs.worker import Worker
from libs.ps import PS
from libs.byz_worker import ByzWorker
from libs import tools

from rsrcs.aggregator_tf.aggregator import Aggregator_tf

from rsrcs.network import Network

# Allowing visualization of the log while the process is running over ssh
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


FLAGS = None


def main():
    n = Network(FLAGS.config)

    if n.get_task_type() == 'worker':
        if n.get_my_attack() != 'None':
            w = ByzWorker(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        else:
            w = Worker(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        w.start()
        model_aggregator = Aggregator_tf(n.get_model_strategy(), len(n.get_all_workers()), FLAGS.nbbyzwrks, FLAGS.native)

        for iter in range(FLAGS.max_iter+1):
            models = w.get_models(iter)
            aggregated_model = model_aggregator.aggregate(models)
            w.write_model(aggregated_model)
            loss, grads = w.compute_gradients(iter)
            print("this is gradeint" , grads)
            w.commit_gradients(grads)

        w.stop(1)

    elif n.get_task_type() == 'ps':
        p = PS(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        p.start()

        gradient_aggregator = Aggregator_tf(n.get_gradient_strategy(), len(n.get_all_workers()), FLAGS.nbbyzwrks, FLAGS.native)

        accuracy = 0
        for iter in range(FLAGS.max_iter):
            models = p.get_models(iter)
            p.write_model(models[0])
            gradients = p.get_gradients(iter)
            print("this is gradient")
            aggregated_gradient = gradient_aggregator.aggregate(gradients)
            model = p.upate_model(aggregated_gradient)
            p.commit_model(model)
            
            tools.training_progression(FLAGS.max_iter, iter, accuracy)
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
