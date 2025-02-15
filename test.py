import replicate
import matplotlib.pyplot as plt
from time import perf_counter

with open("amhc.wav", "rb") as audio_file:
    output = replicate.run(
    "neurotechanalytics/speechcare:a9468353cb7e9c47ee573e3212b4427fdab1cef1cc170920e6ed0125dd7c26ab",
    input={
        "age": 63,
        "mode": "explain_speech",
        "audio": audio_file,
        }
    )

plt.imshow(output)