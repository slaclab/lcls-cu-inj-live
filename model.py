from lume_model.keras import KerasModel
from tensorflow import keras
import numpy as np

# to lume-keras
class ScaleLayer(keras.layers.Layer):
    
    trainable = False
    
    def __init__(self, offset, scale, lower,upper, **kwargs): 
        super(ScaleLayer, self).__init__(**kwargs) 
        self.scale = scale
        self.offset = offset
        self.lower = lower
        self.upper = upper
        
    def call(self, inputs):
        return self.lower+((inputs-self.offset)*(self.upper-self.lower)/self.scale)
    
    
    # MUST OVERRIDE IN ORDER TO SAVE + LOAD W/KERAS
    # STORE SALE, etc.
    def get_config(self):
        return {'scale': self.scale,'offset': self.offset,'lower': self.lower,'upper': self.upper}

# to lume-keras
class UnScaleLayer(keras.layers.Layer):
    
    trainable = False
    
    def __init__(self, offset, scale, lower,upper, **kwargs): 
        super(UnScaleLayer, self).__init__(**kwargs) 
        self.scale = scale
        self.offset = offset
        self.lower = lower
        self.upper = upper
        
    def call(self, inputs):
        return (((inputs-self.lower)*self.scale)/(self.upper-self.lower)) + self.offset
    
    
    # MUST OVERRIDE IN ORDER TO SAVE + LOAD W/KERAS
    # STORE SALE, etc.
    def get_config(self):
        return {'scale': self.scale,'offset': self.offset,'lower': self.lower,'upper': self.upper}

# To lume-keras
class UnScaleImg(keras.layers.Layer):
    
    trainable = False
    
    def __init__(self, img_offset, img_scale, **kwargs): 
        super(UnScaleImg, self).__init__(**kwargs) 
        self.img_scale = img_scale
        self.img_offset = img_offset
        
    def call(self, inputs):
        return (inputs+self.img_offset)*self.img_scale  
    
    
    # MUST OVERRIDE IN ORDER TO SAVE + LOAD W/KERAS
    # STORE SALE, etc.
    def get_config(self):
        return {'img_scale': self.img_scale,'img_offset': self.img_offset}



class FormattedKerasModel(KerasModel):
    def format_input(self, input_dictionary):
        scalar_inputs = np.array([
            input_dictionary['distgen:r_dist:sigma_xy:value'],
            input_dictionary['distgen:t_dist:length:value'],
            input_dictionary['distgen:total_charge:value'],
            input_dictionary['SOL1:solenoid_field_scale'],
            input_dictionary['CQ01:b1_gradient'],
            input_dictionary['SQ01:b1_gradient'],
            input_dictionary['L0A_phase:dtheta0_deg'],
            input_dictionary['L0A_scale:voltage'],
            input_dictionary['end_mean_z']
            ]).reshape((1,9))


        model_input = [scalar_inputs]
        return  model_input


    def parse_output(self, model_output):        
        parsed_output = {}
        parsed_output["x:y"] = model_output[0][0].reshape((50,50))

        # NTND array attributes MUST BE FLOAT 64!!!! np.float() should be moved to lume-epics
        parsed_output["out_xmin"] = np.float64(model_output[1][0][0])
        parsed_output["out_xmax"] = np.float64(model_output[1][0][1])
        parsed_output["out_ymin"] = np.float64(model_output[1][0][2])
        parsed_output["out_ymax"] = np.float64(model_output[1][0][3])

        parsed_output.update(dict(zip(self.output_variables.keys(), model_output[2][0].T)))

        return parsed_output
