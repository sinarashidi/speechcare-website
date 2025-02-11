import torch
import gradio as gr
from replicate.tbnet import TBNet, Config

def predict(audio, age):
    
    config = Config()
    config.seed = 133
    config.bs = 4
    config.epochs = 14
    config.lr = 1e-6
    config.hidden_size = 128
    config.wd = 1e-3
    config.integration = 16
    config.num_labels = 3
    config.txt_transformer_chp = config.MGTEBASE
    config.speech_transformer_chp = config.mHuBERT
    config.segment_size = 5
    config.active_layers = 12
    config.demography = 'age_bin'
    config.demography_hidden_size = 128
    config.max_num_segments = 7

    tbnet_model = TBNet(config)
    tbnet_model.load_state_dict(torch.load("tbnet-best.pt"))
    tbnet_model.eval()
    
    predictions = tbnet_model.inference(audio, age, config)
    return predictions


with gr.Blocks() as demo:
    gr.Markdown("# Cognitive Impairment Prediction from Speech")

    with gr.Row():
        with gr.Column():
            age = gr.Textbox(label="Age", placeholder="Enter your age")
            # Audio input (mic or file upload)
            audio_input = gr.Audio(
                label="Upload or Record Audio",
                type="filepath",  # Using 'numpy' to handle both mic and file uploads
                sources=['microphone', 'upload'],
                max_length=31,
            )
            # Button to trigger prediction
            predict_button = gr.Button("Predict")

        with gr.Column():
            output_text = gr.Markdown(label="Prediction Result", container=True, value="## Prediction Results")
            
    # Link the button to the prediction function
    def update_ui(audio, age):
        if audio is None or age == "":
            return "Please provide both audio and age."

        try:
            # Get predictions from the model
            label, probabilities = predict(audio, int(age))
            labels = {0: "Healthy", 1: "MCI", 2: "ADRD"}
            predicted_label = labels[label]
            max_probability = max(probabilities) * 100  # Convert to percentage
            formatted_output = f"<h2>Predicted Label: {predicted_label}<br>Probability: {max_probability:.2f}%</h2>"
            return formatted_output
        except Exception as e:
            return f"Error: {str(e)}"

    predict_button.click(fn=update_ui, inputs=[audio_input, age], outputs=output_text)

# Launch the Gradio app
demo.launch(share=True)
