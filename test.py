# import torch
# import gradio as gr
# import pandas as pd
# from replicate.tbnet import TBNet, Config


# config = Config()
# config.seed = 133
# config.bs = 4
# config.epochs = 14
# config.lr = 1e-6
# config.hidden_size = 128
# config.wd = 1e-3
# config.integration = 16
# config.num_labels = 3
# config.txt_transformer_chp = config.MGTEBASE
# config.speech_transformer_chp = config.mHuBERT
# config.segment_size = 5
# config.active_layers = 12
# config.demography = 'age_bin'
# config.demography_hidden_size = 128
# config.max_num_segments = 7

# tbnet_model = TBNet(config)
# tbnet_model.load_state_dict(torch.load("tbnet-best.pt"))
# tbnet_model.eval()


# age = 63
# audio = '/workspace/speechcare-website/Datasets/LPF_test_audios/amhc.wav'
# # results = tbnet_model.inference(audio, age, config)
# # print(tbnet_model.illustrate_shap_values())
# tbnet_model.calculate_and_visualize_speech_shap(audio, 'result.png')

# import replicate

# with open("amhc.wav", "rb") as audio_file:
#     # Pass the file object directly to the Replicate API
#     output = replicate.run(
#         "neurotechanalytics/speechcare:7074c14f0aadeba240361a372ff58d66407712c72ac1ff4bde5efab3cf82e6a3",
#         input={
#             'audio': audio_file,  # File object
#             'age': 63             # Age input
#         }
#     )
#     print(output)
    
    
from replicate_integration.predict import Predictor

    