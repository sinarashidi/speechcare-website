import gradio as gr
from gradio_utils import *


replicate_repo_path = setup_env("config/config.yaml")

# Main Page Layout
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
            speech_explanation = gr.Plot(visible=False)
            speech_loading_indicator = gr.HTML("Loading the speech explanations. This make take several seconds...", visible=False)
        
    predict_button.click(
        fn=show_loading,  # Show loading screen
        inputs=[audio_input, age],
        outputs=[output_banner, output_barChart, output_message_area, loading_indicator],
        queue=False  # Ensure this step is executed immediately
    ).then(
        fn=lambda audio, age: update_ui(audio, age, replicate_repo_path),
        inputs=[audio_input, age],
        outputs=[output_banner, output_barChart, output_message_area, loading_indicator]
    )
    
    text_expl_btn.click(
        fn=show_loading_for_explanations,
        outputs=[text_explanation, text_loading_indicator],
    ).then(
        fn=lambda audio, age: get_text_explanations(audio, age, replicate_repo_path),
        inputs=[audio_input, age],
        outputs=[text_explanation, text_loading_indicator],
    )
    
    speech_expl_btn.click(
        fn=show_loading_for_explanations,
        outputs=[speech_explanation, speech_loading_indicator],
    ).then(
        fn=lambda audio, age: get_speech_explanations(audio, age, replicate_repo_path),
        inputs=[audio_input, age],
        outputs=[speech_explanation, speech_loading_indicator],
    )
    
# Launch the Gradio app
demo.launch(share=True)
