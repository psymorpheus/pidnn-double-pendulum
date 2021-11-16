import torch
import yaml
import numpy as np
import pandas as pd
import os
import sys

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
    for noise in common_config['NOISE_CONFIGS']:
        noisedir = f'Noise_{int(100*noise)}'
        for filename in common_config['DATA_CONFIGS']:
            path = os.path.join(modeldir, noisedir + '/' + filename.lower())
            try:
                os.makedirs(path, exist_ok = True)
                print("Successfully created '%s'" % (modeldir + '/' + noisedir + '/' + filename.lower()))
                for modelname in common_config['ALL_MODEL_CONFIGS']:
                    modelpath = os.path.join(path, modelname.lower())
                    os.makedirs(modelpath, exist_ok = True)
                    print("Successfully created '%s'" % (path + '/' + modelname.lower()))
            except OSError as error:
                print("'%s' can not be created" % (modeldir + '/' + noisedir + '/' + filename.lower()))
    print('Successfully created all directories!')

# generate_folders()

def generate_all_datasets():
    for active_data_config_name in common_config['DATA_CONFIGS']:
        active_data_config = all_configs[active_data_config_name].copy()
        active_data_config.update(common_config)
        config = active_data_config

        config['datafile'] = config['TRAINFILE']
        config['vx_range'] = np.linspace(config['TRAIN_VX_START'], config['TRAIN_VX_END'], num = config['TRAIN_VX_VALUES'], dtype=np.float32)
        config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TRAIN_ITERATIONS'], step = config['TIMESTEP'])

        if config['DATASET_CACHING'] and os.path.isfile(config['datadir']+config['datafile']):
            print('Skipping ' + config['datadir'] + config['datafile'])
        else:
            if config['toy_data']:
                toy_datagen(config)
            else:
                simulation_datagen(config)
        
        config['datafile'] = config['TESTFILE']
        new_vx_range = []
        for i in range(len(config['vx_range'])-1):
            new_vx_range.append(np.random.uniform(low=config['vx_range'][i], high=config['vx_range'][i+1], size=(config['TESTSET_MULTIPLIER'],)))
        config['vx_range'] = np.array(new_vx_range).reshape((-1,))

        if config['DATASET_CACHING'] and os.path.isfile(config['datadir']+config['datafile']):
            print('Skipping ' + config['datadir'] + config['datafile'])
        else:
            if config['toy_data']:
                toy_datagen(config)
            else:
                simulation_datagen(config)


# generate_all_datasets()
        
def train_all_models():
    for noise in common_config['NOISE_CONFIGS']:
        for active_data_config_name in common_config['DATA_CONFIGS']:
            active_data_config = all_configs[active_data_config_name].copy()
            active_data_config.update(common_config)

            for active_model_config_name in common_config['MODEL_CONFIGS']:
                if common_config['MODEL_CACHING'] and os.path.isfile(f'./Models/Noise_{int(100*noise)}/{active_data_config_name.lower()}/{active_model_config_name.lower()}.pt'):
                    print(f'======================= Skipping ./Models/Noise_{int(100*noise)}/{active_data_config_name.lower()}/{active_model_config_name.lower()}.pt =======================')
                    continue

                active_model_config = all_configs[active_model_config_name].copy()
                active_model_config.update(active_data_config)
                config = active_model_config

                config['datafile'] = config['TRAINFILE']
                config['vx_range'] = np.linspace(config['TRAIN_VX_START'], config['TRAIN_VX_END'], num = config['TRAIN_VX_VALUES'], dtype=np.float32)
                config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*config['TRAIN_ITERATIONS'], step = config['TIMESTEP'])
                config['noise'] = noise
                config['modeldir'] = 'Models/Noise_' + f'{int(100*noise)}/' + active_data_config_name.lower() + '/'

                print(f'======================={active_data_config_name}, {active_model_config_name}, Noise {int(100*noise)}%=======================')
                pidnn_driver(config)

# train_all_models()

def test_all_models():
    for noise in common_config['NOISE_CONFIGS']:
        dicts_testdata = []

        for active_data_config_name in common_config['DATA_CONFIGS']:
            active_data_config = all_configs[active_data_config_name].copy()
            active_data_config.update(common_config)

            dict_testdata = dict({})

            for active_model_config_name in common_config['MODEL_CONFIGS']:
                active_model_config = all_configs[active_model_config_name].copy()
                active_model_config.update(active_data_config)
                config = active_model_config

                model = torch.load('Models/Noise_' + f'{int(100*noise)}/{active_data_config_name.lower()}/' + active_model_config_name.lower() + '.pt')
                model.eval()

                dict_testdata[active_model_config_name] = "{:.2e}".format(testloader(config, config['datadir'] + config['TESTFILE'], model).item())
           
            dicts_testdata.append(dict_testdata)
        
        df_testdata = pd.DataFrame(dicts_testdata, index = common_config['DATA_CONFIGS'])
        df_testdata.to_csv('Models/Noise_' + f'{int(100*noise)}/' + 'inferences_testdata.csv')
        df_testdata.to_csv(f'inferences_testdata_noise_{int(100*noise)}.csv')

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
            
