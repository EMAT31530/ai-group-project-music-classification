#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 23:58:16 2021

@author: jayvier
"""

import hyperopt
from hyperopt import hp, fmin, tpe, Trials
import os
import keras.backend as K
from CNN import compile_fit_GenreModel
from CNN_final_test import GenreModel
import pickle
import json
from bson import json_util

space = {
    # Loguniform distribustion to find appropriate learning rate multiplier
    'lr_mult' : hp.loguniform('lr_mult', -0.5, 0.5),
    # Uniform distribustion to find appropriate dropout rate
    'dropout' : hp.uniform('dropout',0.0,0.5),
    # To find the best optimizer
    'optimizer' : hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # l2 weight regularization multiplier
    'l2_mult' : hp.loguniform('l2_mult', -1.3, 1.3),
    # Choice of suitable activations
    'activation' : hp.choice('activation', ['relu', 'elu']),
    # batch normalization or not
    'use_BN' : hp.choice('use_BN', [True, False]),
    # size of the n x n kernel used for each filter in conv layers
    'kern_size' : hp.quniform('kern_size', 2, 4, 1),      # Haven't added to model yet
    # pooling type max or avg
    'pooling' : hp.choice('pooling', ['avg', 'max']),
    # use batch normalization
    'dropout_conv' : hp.choice('dropout_conv', [None, hp.uniform('conv_drop_rate', 0.0, 0.35)])
}


RESULTS_DIR = 'Optimization_results'


def optimize(hype_space):
    
    model, model_name, result = GenreModel(hype_space)
    # Save result
    save_result(model_name, result)
    
    K.clear_session()
    
    return result






def save_result(model_name, result):
    #Save json to a directory and a filename.
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, result_name), 'w') as f:
        json.dump(result,
                  f,
                  default=json_util.default,
                  sort_keys=True,
                  indent=4, separators=(',', ': '))
        
   
        
def load_result(result_name):
    #Load json from a path (directory + filename).
    result_path = os.path.join(RESULTS_DIR, result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
            # default=json_util.default,
            # separators=(',', ': ')
        )

def load_best_hyperspace():
    results = [
        f for f in list(sorted(os.listdir(RESULTS_DIR))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_result(best_result_name)["space"]

def get_best_model():
    results = [f for f in list(sorted(os.listdir(RESULTS_DIR))) if 'json' in f]  # sorted in order of validation accuracies
    
    if len(results) == 0:
        return None
    
    best_result_name = results[-1]
    best_hyperspace = load_result(best_result_name)['space']
    
    return

def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))


def run_a_trial():
    # function that runs a single optimization
    # first we check to see if theres any previous results saved
    max_evals = nb_evals = 1
    try:
        trials = pickle.load(open('results.pkl', 'rb'))
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(len(trials.trials)))
    except:
        trials = Trials()
        print('Starting new trial.')
        
    best = fmin(optimize,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=max_evals,
            )
    
    pickle.dump(trials, open('results.pkl', 'wb'))   # saves trials object as pickled file
    



if __name__ == '__main__':
    # Runs optimization process indefinatley and saves results
    while True:            
        run_a_trial()        




