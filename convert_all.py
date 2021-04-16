import os
from snntoolbox.utils.utils import import_configparser
from snntoolbox.bin.run import main



def mk_config(path,fold):
    "create config file given path to a trained model"
    
    configparser = import_configparser()
    config = configparser.ConfigParser()

    config['paths'] = {
        'path_wd': path+"/"+fold,       # Path to model.
        'dataset_path': path+"/"+fold,  # Path to dataset.
        'filename_ann': "ts"+fold       # Name of input model.
        }
    config['tools'] = {
        'evaluate_ann': True,           # Test ANN on dataset before conversion.
        'normalize':True,               # Normalize weights for full dynamic range.
        'simulate':True                 # Simulate model, seems to be necessary for normalization
    }
    config['simulation'] = {
        'simulator': 'nest',            # We convert to a pynn model with nest as backend
        'duration': 50,                 # Number of time steps to run each sample.
        'num_to_test': 1,               # How many test samples to run.
        'batch_size': 1,                # Batch size for simulation.
        'dt': 0.1                       # timestep
    }
    config['input'] = {
        'poisson_input': True,          # Images are encodes as spike trains.
        'input_rate': 1000              # Poisson Neurons firing rate
    }
    config['cell'] = {                  
        'v_thresh' : 0.01,              # voltage threshold, different for actual SCNN
        'tau_refrac' : 0.1,             # refractory period
        'delay' : 0.1                   # synaptic delay
    }

    return config

    
# loop through all participants
for sub in range(1,10):
    for fold in range(5):
        
        # create config
        config = mk_config("/home/matthijspals/spin/models/subject_"+str(sub),"fold"+str(fold))

        # run snntoolbox
        config_filepath = os.getcwd() + '/config'
        with open(config_filepath, 'w') as configfile:
            config.write(configfile)
    
        main(config_filepath)
