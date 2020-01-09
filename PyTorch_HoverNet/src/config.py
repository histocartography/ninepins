import importlib
import numpy as np



####
class Config(object):
    def __init__(self, ):

        self.seed = 10
        mode = 'hover'
        self.model_type = 'np_hv'

        self.type_classification = True # whether to predict the nuclear type
        # ! must use CoNSeP dataset, where nuclear type labels are available
        self.nr_types = 5  # denotes number of classes for nuclear type classification
        # ! some semantic segmentation network like micronet,
        # ! nr_types will replace nr_classes if type_classification=True
        self.nr_classes = 2 # Nuclei Pixels vs Background

        #### Dynamically setting the config file into variable
        if mode == 'hover':
            config_file = importlib.import_module('opt.hover') # np_hv, np_dist
        else:
            config_file = importlib.import_module('opt.other') # fcn8, dcan, etc.
        config_dict = config_file.__getattribute__(self.model_type)

        self.input_norm  = True # normalize RGB to 0-1 range

        ####
        exp_id = 'v1.0/'
        model_id = '%s' % self.model_type
        #self.model_name = '%s/%s' % (exp_id, model_id)
        # loading chkpts in tensorflow, the path must not contain extra '/'
        self.log_path = '/dataT/frd/hover_net/src/logs' # log root path - modify according to needs
        self.save_dir = '%s/%s' % (self.log_path, self.model_name) # log file destination

        #### Info for running inference

        # file for loading weights
        self.weight_file = '/Users/frd/Documents/Code/CoNSeP/Train/hover_seg_&_class_CoNSeP.npz'

        # output will have channel ordering as [Nuclei Type][Nuclei Pixels][Additional]
        # where [Nuclei Type] will be used for getting the type of each instance
        # while [Nuclei Pixels][Additional] will be used for extracting instances

        self.inf_imgs_ext = '.png'
        self.inf_data_dir = '/dataT/frd/CoNSeP/Test/Images/'
        self.inf_output_dir = '/dataT/frd/hover_net/src/output/%s/%s/' % (exp_id, model_id)

        # for inference during evalutaion mode i.e run by infer.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
        # for inference during training mode i.e run by trainer.py
        #self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']

    def get_model(self):
        if self.model_type == 'np_hv':
            model_constructor = importlib.import_module('model.graph')
            model_constructor = model_constructor.Model_NP_HV #in graph.py
        elif self.model_type == 'np_dist':
            model_constructor = importlib.import_module('model.graph')
            model_constructor = model_constructor.Model_NP_DIST
        else:
            model_constructor = importlib.import_module('model.%s' % self.model_type)
            model_constructor = model_constructor.Graph
        return model_constructor # NOTE return alias, not object
