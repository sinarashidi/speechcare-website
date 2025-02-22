import requests
import json


text =  "more on what I should say or what I'm saying or try to think more. Slow down and tell you something that happened to me that was very good. A couple years ago I was going to work and I was about to make a left turn and I kind of got didn't remember how to make the left turn."

dic = {' ': 0.014079879969358444,
 'more ': 0.005503218621015549,
 'on ': 0.021756412461400032,
 'what ': 0.003668581135571003,
 'I  should ': -0.016871580854058266,
 'say ': -0.014560109004378319,
 'or ': -0.012975558638572693,
 "I '": -0.0014825304970145226,
 'm ': -0.01105587650090456,
 'saying ': -0.014675767160952091,
 'try ': -0.006820406764745712,
 'to ': -0.00657994206994772,
 'think ': 0.017855364829301834,
 'more': 0.029153011739253998,
 '. ': 0.0,
 'Slow ': -0.03418763354420662,
 'down ': 0.028165090829133987,
 'and ': 0.035104503855109215,
 'tell  you ': 0.006873702630400658,
 'something ': 0.014232566580176353,
 'that ': 0.01739544700831175,
 'happened ': -0.012668331153690815,
 'me ': 0.019205288030207157,
 'was ': -0.03241345100104809,
 'very ': -0.013372650370001793,
 'good': 0.02296769618988037,
 'A  couple ': 0.0014515221118927002,
 'years  ago ': -0.009059272706508636,
 'I ': -0.019976818934082985,
 'going  to ': -0.021259513683617115,
 'work ': -0.02476880233734846,
 'I  was ': -0.03414250444620848,
 'about  to ': -0.010339449159801006,
 'make  a ': 0.0011140396818518639,
 'left  turn ': -0.00013471487909555435,
 'I  kind ': -0.04249691218137741,
 'of  got ': -0.06698150187730789,
 "didn '": -0.005156556144356728,
 't ': -0.023289134725928307,
 'remember  how ': 0.0003440435975790024,
 'to  make ': -0.014750635251402855,
 'the  left ': 0.02786958869546652,
 'turn': 0.02657183911651373,
 '.': 0.06609720923006535,
 '': 3.5762786865234375e-07}


message = f"""
You are a specialized language model trained to detect linguistic cues of cognitive impairment. You will receive:
1) A set of linguistic features to consider.
2) A text passage to analyze.
3) Token-level SHAP values from a pre-trained model.

Your task is to:
A. Identify which tokens are most influential in the classification, based on SHAP values.
B. Map each influential token to one or more linguistic features (e.g., lexical richness, syntactic complexity).
C. Explain how the token and its context may indicate healthy cognition or cognitive impairment.

Please follow the steps below and provide a structured analysis.
---
Linguistic Features to Consider:
• **Lexical Richness**: Unusual or varied vocabulary, overuse of vague terms (e.g., “thing,” “stuff”).
• **Syntactic Complexity**: Simple vs. complex sentence constructions, grammatical errors.
• **Sentence Length and Structure**: Fragmented vs. compound/complex sentences.
• **Repetition**: Repeated words, phrases, or clauses.
• **Disfluencies and Fillers**: Terms like “um,” “uh,” “like.”
• **Semantic Coherence and Content**: Logical flow of ideas, clarity of meaning.
• **Additional Feature (XXX)**: Placeholder for any extra marker (e.g., specialized domain terms).
---
Text to Analyze:
{text}
---
Token-level SHAP Values:
{dic}
---
Analysis Format:
1) **Token-Level Analysis**: For each token with a significant |SHAP| value, specify:
   - Token
   - SHAP Value
   - Linguistic Feature(s) Involved
   - Brief Interpretation
2) **Overall Summary**: Synthesize the significance of these tokens/features to explain how they collectively point to healthy cognition or potential cognitive impairment.
---
"""
output_format = """
Example Output Format:
{
  "Analysis": [
    {
      "Token": "um",
      "SHAP_Value": -0.87,
      "Linguistic_Feature": "Disfluency",
      "Interpretation": "Filler word commonly seen in cognitive impairment contexts."
    },
    {
      "Token": "patient",
      "SHAP_Value": 0.65,
      "Linguistic_Feature": "Lexical Richness",
      "Interpretation": "Use of domain-specific term suggests context awareness."
    },
    ...
  ],
  [sep_token]
  "Overall_Summary": "Multiple disfluencies and repetitive fragments are indicative of possible cognitive impairment."
  [sep_token]
}
---
Constraints and Guidelines:
- Rely only on the provided text and SHAP values; do not infer from external or hidden knowledge.
- Tie each token with high |SHAP| back to a specific linguistic feature and explain its clinical relevance.
- Put separator token [sep_token] before and after the 'Overall_Summary' in the output.
"""


message += output_format

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "meta-llama/llama-3.3-70b-instruct:free",
    "messages": [
      {
        "role": "user",
        "content": message
      }
    ],
    
  })
)

r = response.json()['choices'][0]['message']['content']

with open("test.txt", 'w') as f:
  f.write(r)
  
with open('test.txt', 'a') as f:
  summary = r.split('[sep_token]')[1].split(':')[1].strip()
  f.write('\n\n\n')
  f.write(summary)