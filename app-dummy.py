import torch
import gradio as gr
import pandas as pd
# from replicate.tbnet import TBNet, Config
from time import sleep

    
    
def predict(audio, age):
    sleep(2)
    label , probs = "2", [0.1, 0.2, 0.7]
    output = pd.DataFrame(probs, columns=["Probability"], index=["Healthy", "MCI", "ADRD"])
    return label, output


def get_text_explanations():
    sleep(2)
    text_shap_html = "<h2>Text Explanations</h2>"
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
        output_message = f"You are {cog_imp_percent:.2f}% at risk of cognitive impairment."

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
    gr.HTML("SpeechCare", elem_classes=["page-title"], padding=False)
    gr.HTML("Cognitive Impairment Detection from Speech + Explainability in both Text and Speech Modalities", elem_classes=["page-subtitle"], padding=False)
    gr.HTML('<div class="page-header-line"></div>', padding=False)
    gr.HTML("Cognitive Impairment Detection", elem_classes=["section-header"])
    
    with gr.Row():
        # Choose language
        with gr.Column():
            gr.HTML("1. Select Your Language", elem_classes=["instruction"], padding=False)
            language = gr.Dropdown(["Select Language", "English", "Spanish"], label="Language", interactive=True, value="Select Language")
        with gr.Column():
            gr.HTML("2. Enter Your Age", elem_classes=["instruction"], padding=False)
            age = gr.Textbox(label="Age", placeholder="Enter your age", elem_classes=["age-input"])
            
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
    gr.HTML("4. Record or Upload Audio", elem_classes=["instruction"], padding=False)
    with gr.Row(elem_classes=["audio-input-row"]):
        with gr.Column(scale=3):
            audio_input = gr.Audio(
                label="Audio",
                type="filepath",
                sources=['microphone', 'upload'],
                max_length=31,
                elem_classes=["audio-input"],
            )
    with gr.Row(elem_classes=["predict-row"]):
        predict_button = gr.Button("Predict", elem_classes=["predict-btn"])

    output_banner = gr.HTML("<h2>Prediction Results</h2>", visible=False, elem_classes=['prediction-banner'])
    output_barChart = gr.BarPlot(
        x="Label", 
        y="Probability",
        sort=None,
        orientation="h",
        visible=False,
        elem_classes=["output-barchart"],
        container=False,
        y_lim=[0.0, 1.0],
        height=280,
    )
    output_message_area = gr.HTML("", visible=False)
    loading_indicator = gr.HTML("Loading...", visible=False)

            
    # Explainability Section
    gr.HTML('<div class="horizontal-line"></div>')
    gr.HTML("Explainability", elem_classes=["section-header"])
    with gr.Row(min_height=150, elem_classes=["explanation-row"]):
        with gr.Column(scale=1):
            text_expl_btn = gr.Button("Calculate Text SHAP Results", elem_classes=["text-expl-btn"])
        with gr.Column(scale=3):
            text_explanation = gr.HTML("", visible=False)
            text_loading_indicator = gr.HTML("Loading the text explanations. This may take up to 2 minutes...", visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            speech_expl_btn = gr.Button("Calculate Speech SHAP Results", elem_classes=["speech-expl-btn"])
        with gr.Column(scale=3):
            speech_explanation = gr.HTML("", visible=False)
            speech_loading_indicator = gr.HTML("Loading the speech explanations. This make take several seconds...", visible=False)
        
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
