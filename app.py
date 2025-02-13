import torch
import gradio as gr
import pandas as pd
from replicate.tbnet import TBNet, Config


def predict(audio, age):
    
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
    
    # predictions = tbnet_model.inference(audio, age, config)
    # return predictions
    probs = [0.5, 0.3, 0.2]
    output = pd.DataFrame(probs, columns=["Probability"], index=["Healthy", "MCI", "ADRD"])
    return output.idxmax(), output


with gr.Blocks(css_paths='styles.css', theme='ocean') as demo:
    gr.Markdown("# SpeechCare")
    gr.Markdown(" ## Cognitive Impairment Detection from Speech")

    with gr.Row():
        with gr.Column():
            age = gr.Textbox(label="Age", placeholder="Enter your age")
            # Audio input (mic or file upload)
            audio_input = gr.Audio(
                label="Upload or Record Audio",
                type="filepath",
                sources=['microphone', 'upload'],
                max_length=31,
            )
            # Button to trigger prediction
            predict_button = gr.Button("Predict")

        with gr.Column():
            output_banner = gr.Markdown("## Prediction Results", visible=False)
            output_barChart = gr.BarPlot(
                x="Label", 
                y="Probability",
                sort=None,
                orientation="h",
                visible=False,
            )
            output_message_area = gr.HTML("", elem_classes=["output-message"], visible=False)
    # Link the button to the prediction function
    def update_ui(audio, age):
        if not audio or age.strip() == "":
            # Return an empty BarPlot in case of missing input
            empty_df = pd.DataFrame({"Label": [], "Probability": []})
            return gr.BarPlot(value=empty_df, x="Label", y="Probability")
        
        try:
            label, probabilities = predict(audio, int(age))
            df = probabilities.reset_index().rename(columns={"index": "Label"})
            # Sum of the probabilities of MCI and ADRD
            cog_imp_percent = df.loc[1:, "Probability"].sum() * 100
            output_message = f"You are {cog_imp_percent}% in risk of cognitive impairment."
            return (
                gr.update(visible=True),
                gr.BarPlot(value=df, x="Label", y="Probability", sort=["Healthy", "MCI", "ADRD"], orientation="h", visible=True),
                gr.HTML(output_message, visible=True)
            )
        except Exception as e:
            # Return an error message as a Markdown component
            return gr.Markdown(f"Error: {str(e)}")
        
    predict_button.click(
        fn=update_ui, 
        inputs=[audio_input, age], 
        outputs=[output_banner, output_barChart, output_message_area]
    )

# Launch the Gradio app
demo.launch(debug=True)
