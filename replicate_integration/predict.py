import torch
from replicate_integration.tbnet import TBNet, Config
from cog import BasePredictor, Path, Input, File
 
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        self.config = Config()
        self.config.seed = 133
        self.config.bs = 4
        self.config.epochs = 14
        self.config.lr = 1e-6
        self.config.hidden_size = 128
        self.config.wd = 1e-3
        self.config.integration = 16
        self.config.num_labels = 3
        self.config.txt_transformer_chp = self.config.MGTEBASE
        self.config.speech_transformer_chp = self.config.mHuBERT
        self.config.segment_size = 5
        self.config.active_layers = 12
        self.config.demography = 'age_bin'
        self.config.demography_hidden_size = 128
        self.config.max_num_segments = 7
        
        self.net = TBNet(self.config)
        self.net.load_state_dict(torch.load("tbnet-best.pt")) # Modify the path if needed
        self.net.eval()



    def predict(self, 
            audio: File = Input(description="Input Speech"),
            age: float = Input(description="Age"),
            ) -> Path:
        """Run a single prediction on the model"""
        predictions = self.net.inference(audio, age, self.config)
        return predictions