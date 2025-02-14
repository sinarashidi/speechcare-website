import torch
import gradio as gr
import pandas as pd
from replicate.tbnet import TBNet, Config
from time import sleep


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
    
    
def predict(audio, age):
    label, probs = tbnet_model.inference(audio, age, config)
    output = pd.DataFrame(probs, columns=["Probability"], index=["Healthy", "MCI", "ADRD"])
    return label, output


def get_text_explanations():
    text_shap_html = tbnet_model.illustrate_shap_values()
    return (
        gr.HTML(text_shap_html, visible=True),
        gr.update(visible=False), 
    )


def show_loading(audio, age):
    if not audio or age.strip() == "":
        return (
            gr.update(visible=False),
            gr.BarPlot(visible=False),
            gr.HTML(visible=False),
            gr.update(visible=False)
        )

    return (
        gr.update(visible=False),  # Hide output_banner
        gr.update(visible=False),  # Hide output_barChart
        gr.update(visible=False),  # Hide output_message_area
        gr.update(visible=True)
    )
    

def show_loading_for_text_explanations():
    return (
        gr.update(visible=False),
        gr.update(visible=True)
    )
    
    
def update_instructions(language):
        if language == "Spanish":
            return (
                    gr.HTML("3. Read the Following Sentence Clearly:", elem_classes=["instruction"], padding=False, visible=True),
                    gr.HTML("Por favor, lee el texto a continuación en tu voz normal. Cuando estés listo, haz clic en el botón Grabar para comenzar a leer. Si ya has grabado tu respuesta, puedes subir tu archivo de audio en su lugar.", visible=True), 
                    gr.HTML(value="""<h2 style="text-align: center;">En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, 
                            adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, 
                            lentejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda.</h2>""", container=True, visible=True),
                    gr.update(visible=False)
                )
        elif language == "English":
            return (
                    gr.HTML("3. Describe the Picture Below", elem_classes=["instruction"], padding=False, visible=True),
                    gr.HTML("Please take a good look at the picture below and describe everything that you see going on in that picture. When you are ready, click the Record button to start speaking. If you have already recorded your response, you can upload your audio file instead.", visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    

def update_ui(audio, age):
    if not audio or age.strip() == "":
        return (
            gr.update(visible=False),
            gr.BarPlot(visible=False),
            gr.HTML("Please enter both age and audio", visible=True, elem_classes=["error-message"]),
            gr.update(visible=False)
        )

    try:
        _, probabilities = predict(audio, int(age))
        df = probabilities.reset_index().rename(columns={"index": "Label"})
        # Sum of the probabilities of MCI and ADRD
        cog_imp_percent = df.loc[1:, "Probability"].sum() * 100
        output_message = f"You are {cog_imp_percent:.2f}% in risk of cognitive impairment."
        return (
            gr.update(visible=True),
            gr.BarPlot(value=df, x="Label", y="Probability", sort=["Healthy", "MCI", "ADRD"], orientation="h", visible=True),
            gr.HTML(output_message, elem_classes=["output-message"] , visible=True),
            gr.update(visible=False)
        )
    except Exception as e:
        return (
            gr.update(visible=False),
            gr.BarPlot(visible=False),
            gr.HTML(str(e), visible=True, elem_classes=["error-message"]),
            gr.update(visible=False)
        )

#    
### Page Layout
#
with gr.Blocks(css_paths='styles.css', theme='ocean') as demo:
    gr.Markdown("# SpeechCare")
    gr.Markdown(" ## Cognitive Impairment Detection from Speech + Explainability in both Text and Audio Modalities")
    gr.HTML('<div class="page-header-line"></div>')
    gr.HTML("Cognitive Impairment Detection", elem_classes=["section-header"])
    
    with gr.Row():
        with gr.Column():
            gr.HTML("1. Enter Your Age", elem_classes=["instruction"], padding=False)
            age = gr.Textbox(label="Age", placeholder="Enter your age", elem_classes=["age-input"])
        # Choose language
        with gr.Column():
            gr.HTML("2. Select Your Language", elem_classes=["instruction"], padding=False)
            language = gr.Dropdown(["Select Language", "English", "Spanish"], label="Language", interactive=True, value="Select Language")
            
    with gr.Column():
        # English instructions when the language is set to English
        title_display = gr.HTML(visible=False)
        english_description_display = gr.HTML(visible=False)

        # Will be updated dynamically when Spanish is selected
        spanish_text_display = gr.HTML(visible=False)

        image_display = gr.Image("images/cookie_theft_picture.jpg", 
                                container=False,
                                show_download_button=False, 
                                show_fullscreen_button= False,
                                elem_classes=["cookie-theft-image"],
                                visible=False,
                                )

    language.change(update_instructions, inputs=language, outputs=[title_display, english_description_display, spanish_text_display, image_display])
        
    
    with gr.Row(elem_classes=["audio-input-row"]):
        with gr.Column():
            gr.HTML("4. Record or Upload Audio", elem_classes=["instruction"], padding=False)
            audio_input = gr.Audio(
                label="Audio",
                type="filepath",
                sources=['microphone', 'upload'],
                max_length=31,
                elem_classes=["audio-input"],
            )

    with gr.Row(elem_classes=["predict-row"]):
        predict_button = gr.Button("Predict", elem_classes=["predict-btn"])

    with gr.Row():
        output_banner = gr.Markdown("## Prediction Results", visible=False)
        output_barChart = gr.BarPlot(
            x="Label", 
            y="Probability",
            sort=None,
            orientation="h",
            visible=False,
        )
        output_message_area = gr.HTML("", elem_classes=["output-message"], visible=False)
        loading_indicator = gr.HTML("<div class=inference-loading>Loading...<img src=\"images/loading.gif\" class=loading-image></div>", visible=False)

            
    # Explainability Section
    gr.HTML('<div class="horizontal-line"></div>')
    gr.HTML("Explainability", elem_classes=["section-header"])
    with gr.Row():
        with gr.Column(scale=1):
            text_expl_btn = gr.Button("Text Explanability", elem_classes=["text-expl-btn"])
        with gr.Column(scale=3):
            text_explanation = gr.HTML("", visible=False)
            text_loading_indicator = gr.HTML("Loading the text explanations. This make take up to 2 minutes...", visible=False)
    # Link the button to the prediction function
        
        
    predict_button.click(
        fn=show_loading,  # Show loading screen
        inputs=[audio_input, age],
        outputs=[output_banner, output_barChart, output_message_area, loading_indicator],
        queue=False  # Ensure this step is executed immediately
    ).then(
        fn=update_ui,  # Run the prediction and update the UI
        inputs=[audio_input, age],
        outputs=[output_banner, output_barChart, output_message_area, loading_indicator]
    )
    
    text_expl_btn.click(
        fn=show_loading_for_text_explanations,
        outputs=[text_explanation, text_loading_indicator],
    ).then(
        fn=get_text_explanations,
        outputs=[text_explanation, text_loading_indicator],
    )
# Launch the Gradio app
demo.launch(share=True)
