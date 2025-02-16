import os
import requests
import yaml
import replicate
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def setup_env(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    os.environ["REPLICATE_API_TOKEN"] = config["replicate_api_token"]
    repo_path = config["replicate_model_repo"]
    return repo_path


def predict(audio, age, repo_path):
    with open(audio, "rb") as audio_file:
        output = replicate.run(
        repo_path,
        input={
            "age": int(age),
            "mode": "inference",
            "audio": audio_file,
            }
        )
    label, probs = output
    output = pd.DataFrame(probs, columns=["Probability"], index=["Healthy", "MCI", "ADRD"])
    return label, output


def get_text_explanations(audio, age, repo_path):
    with open(audio, "rb") as audio_file:
        text_shap_html = replicate.run(
        repo_path,
        input={
            "age": int(age),
            "mode": "explain_text",
            "audio": audio_file,
            }
        )
    return (
        gr.HTML(text_shap_html, visible=True),
        gr.update(visible=False), 
        gr.update(visible=True)
    )
    
    
def get_llama_explanations(audio, age, repo_path):
    with open(audio, "rb") as audio_file:
        llama_interprets = replicate.run(
            repo_path,
            input={
                "age": int(age),
                "mode": "llama",
                "audio": audio_file,
            }
        )
    return (
        gr.Markdown(llama_interprets, visible=True),
        gr.update(visible=False)
    )


def get_speech_explanations(audio, age, repo_path):
    with open(audio, "rb") as audio_file:
        image_url = replicate.run(
            repo_path,
            input={
                "age": int(age),
                "mode": "explain_speech",
                "audio": audio_file,
            }
        )
    # Download the image from the URL
    response = requests.get(image_url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    # Return the image and update the loading indicator
    return (
        gr.Image(img, visible=True), 
        gr.update(visible=False)
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

    
def show_loading_for_explanations():
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


def update_ui(audio, age, repo_path):
    if not audio or age.strip() == "":
        return (
            gr.update(visible=False),
            gr.BarPlot(visible=False),
            gr.HTML("Please enter both age and audio", visible=True, elem_classes=["error-message"]),
            gr.update(visible=False)
        )

    try:
        _, probabilities = predict(audio, age, repo_path)
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
