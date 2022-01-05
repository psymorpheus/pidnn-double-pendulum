import torch
import yaml
import numpy as np
import pandas as pd
import os
import sys
from dp_datagen import double_pendulum_data
from dp_pidnn import pidnn_driver
from dp_dataloader import testloader
from dp_ff import ff_driver

if len(sys.argv) > 2:
    config_filename = sys.argv[2]
else:
    config_filename = "dp_config.yaml"

with open(config_filename, "r") as f:
    all_configs = yaml.safe_load(f)
    common_config = all_configs['COMMON'].copy()
    # Filling in models from model templates
    for instance in common_config['MODEL_CONFIGS']:
        template_name = instance[:instance.rfind('_')]
        training_points = int(instance[(instance.rfind('_')+1):])
        template_config = all_configs[template_name].copy()
        template_config['num_datadriven'] = training_points
        # if template_config['num_collocation'] == -1: template_config['num_collocation'] = 10 * training_points
        template_config['model_name'] = template_name.lower() + '_' + str(training_points)
        all_configs[template_name + '_' + str(training_points)] = template_config

def generate_folders():
    datadir = './Data'
    for filename in common_config['DATA_CONFIGS']:
        path = os.path.join(datadir, filename.lower())
        try:
            os.makedirs(path, exist_ok = True)
            print("Successfully created '%s'" % (datadir+'/'+filename.lower()))
        except OSError as error:
            print("'%s' can not be created" % (datadir+'/'+filename.lower()))
    modeldir = './Models'
    for seed in common_config['SEEDS']:
        seeddir = f'SEED_{seed}'
        for noise in common_config['NOISE_CONFIGS']:
            noisedir = f'Noise_{int(100*noise)}'
            for filename in common_config['DATA_CONFIGS']:
                path = os.path.join(modeldir, seeddir + '/' + noisedir + '/' + filename.lower())
                try:
                    os.makedirs(path, exist_ok = True)
                    print("Successfully created '%s'" % (modeldir + '/' + seeddir + '/' + noisedir + '/' + filename.lower()))
                    for modelname in common_config['ALL_MODEL_CONFIGS']:
                        modelpath = os.path.join(path, modelname.lower())
                        os.makedirs(modelpath, exist_ok = True)
                        print("Successfully created '%s'" % (path + '/' + modelname.lower()))
                except OSError as error:
                    print("'%s' can not be created" % (modeldir + '/' + noisedir + '/' + filename.lower()))
    print('Successfully created all directories!')

def generate_all_datasets():
    for active_data_config_name in common_config['DATA_CONFIGS']:
        active_data_config = all_configs[active_data_config_name].copy()
        active_data_config.update(common_config)
        config = active_data_config

        config['datafile'] = config['TRAINFILE']
        config['theta_range'] = np.linspace(config['TRAIN_THETA_START']*np.pi/180.0, config['TRAIN_THETA_END']*np.pi/180.0, num = config['TRAIN_THETA_VALUES'], dtype=np.float32)
        config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TRAIN_ITERATIONS'], step = config['TIMESTEP'])

        if config['DATASET_CACHING'] and os.path.isfile(config['datadir']+config['datafile']):
            print('Skipping ' + config['datadir'] + config['datafile'])
        else:
            double_pendulum_data(config)
        
        config['datafile'] = config['TESTFILE']
        new_theta_range = []
        for i in range(len(config['theta_range'])-1):
            new_theta_range.append(np.random.uniform(low=config['theta_range'][i], high=config['theta_range'][i+1], size=(config['TESTSET_MULTIPLIER'],)))
        config['theta_range'] = np.array(new_theta_range).reshape((-1,))

        if config['DATASET_CACHING'] and os.path.isfile(config['datadir']+config['datafile']):
            print('Skipping ' + config['datadir'] + config['datafile'])
        else:
            double_pendulum_data(config)


# generate_all_datasets()
        
def train_all_models():
    for seed in common_config['SEEDS']:
        for noise in common_config['NOISE_CONFIGS']:
            for active_data_config_name in common_config['DATA_CONFIGS']:
                active_data_config = all_configs[active_data_config_name].copy()
                active_data_config.update(common_config)

                for active_model_config_name in common_config['MODEL_CONFIGS']:
                    if common_config['MODEL_CACHING'] and os.path.isfile(f'./Models/SEED_{seed}/Noise_{int(100*noise)}/{active_data_config_name.lower()}/{active_model_config_name.lower()}.pt'):
                        print(f'======================= Skipping ./Models/SEED_{seed}/Noise_{int(100*noise)}/{active_data_config_name.lower()}/{active_model_config_name.lower()}.pt =======================')
                        continue

                    active_model_config = all_configs[active_model_config_name].copy()
                    active_model_config.update(active_data_config)
                    config = active_model_config

                    config['datafile'] = config['TRAINFILE']
                    config['noise'] = noise
                    config['seed'] = seed
                    config['modeldir'] = 'Models/' + f'SEED_{seed}/' + f'Noise_{int(100*noise)}/' + active_data_config_name.lower() + '/'

                    print(f'======================={active_data_config_name}, {active_model_config_name}, Noise {int(100*noise)}%=======================')
                    if config['take_differential_points']:
                        pidnn_driver(config)
                    else:
                        ff_driver(config)

# train_all_models()

def test_all_models():
    dicts_testdata = []
    for noise in common_config['NOISE_CONFIGS']:
        for active_data_config_name in common_config['DATA_CONFIGS']:
            active_data_config = all_configs[active_data_config_name].copy()
            active_data_config.update(common_config)

            for active_model_config_name in common_config['MODEL_CONFIGS']:
                active_model_config = all_configs[active_model_config_name].copy()
                active_model_config.update(active_data_config)
                config = active_model_config

                seed_results = []
                for seed in common_config['SEEDS']:
                    model = torch.load(f'Models/SEED_{seed}/Noise_' + f'{int(100*noise)}/{active_data_config_name.lower()}/' + active_model_config_name.lower() + '.pt')
                    model.eval()
                    seed_results.append(testloader(config, config['datadir'] + config['TESTFILE'], model).item())
                seed_results = np.array(seed_results)

                result = (noise,
                    active_data_config_name,
                    active_model_config_name,
                    np.mean(seed_results),
                    np.std(seed_results),
                    np.max(seed_results),
                    np.min(seed_results),
                    np.std(-np.log(seed_results))
                    )
                print(result)
                dicts_testdata.append(result)
            
    df_testdata = pd.DataFrame(dicts_testdata,\
        columns=['NOISE', 'DATASET', 'MODEL', 'ERR_AVG', 'ERR_STD', 'ERR_MAX', 'ERR_MIN', 'LOG_STD'])
    df_testdata.to_csv(f'Inferences/inferences.csv')

# test_all_models()

if __name__=="__main__":
    command = sys.argv[1]
    if command == 'folders':
        generate_folders()
    elif command == 'datasets':
        generate_all_datasets()
    elif command == 'train':
        train_all_models()
    elif command == 'test':
        test_all_models()
    else:
        print('Please input valid keyword')
        sys.exit(1)
            
