# Evaluating-Prompting-and-Fine-Tuning-Strategies

# Appendix A - Metric Glossary

This table provides a non-technical overview of the metrics used in this study to evaluate the performance of prompting versus fine-tuning strategies.

| Metric | Technical Name | Pedagogical "Plain English" Meaning | Goal in this Study |
| :--- | :--- | :--- | :--- |
| **GLEU** | Google-BLEU | **Fidelity/Minimalism:** Measures how much of the student's original voice was preserved. | **Higher is better.** A high score indicates the model avoided unnecessary rewriting. |
| **PPL** | Perplexity | **Fluency/Naturalness:** Measures how much the correction sounds like a natural English speaker. | **Lower is better.** High scores indicate the AI produced "gibberish" or awkward phrasing. |
| **$F_{0.5}$** | F-Score ($\beta=0.5$) | **Precision/Reliability:** Ensures that when the AI makes a change, it is a correct and necessary one. | **Higher is better.** This weights precision over recall to penalize "over-correction." |
| **ERRANT** | Error Annotation Toolkit | **Diagnostic Labels:** Categorizes the specific types of errors found (e.g., Verbs, Nouns, Punctuation). | **Categorization.** Used to analyze if the AI is focusing on the correct linguistic categories. |

# Appendix B — Prompt Examples

## Simple Instructions Prompt

For the essay I send you, correct the English.  
- Correct the grammar.  
- Correct the spelling.  
- Make sentences sound natural.  

## Complex Instructions Prompt

**System:** For the provided text, you will correct the English as follows:

1. **First, correct all spelling mistakes.**  
   Do not change spelling between different Englishes such as US English or UK English.  
   Do not mark spelling changes with brackets or anything else.

2. **Do not change grammar between different Englishes**,  
   but fix any obviously wrong sentences (those that are confusing or highly unusual English usage).  
   Do not change any sentences that make sense but are only slightly unusual.  
   Do not substitute words (e.g., “because”/“since”, “but”/“however”, “next”/“afterward”)  
   unless the meaning is wrong.

3. **Join or split sentences when necessary:**  
   - Join sentences when it is more natural.  
   - Split sentences when a sentence is unnaturally long.

4. **Bracket optional words:**  
   - If a word is optional and removing it would *not* make the sentence ungrammatical, put it in brackets.  
   - Do **not** bracket required words.  
   - Only bracket added words if they are optional.

**Example:**  
Input:  
`I wents to hospital tooday.`  
Output:  
`I went to (the) hospital today.`  
“the” is optional → bracketed.

## Additional Example

**Input:**  
It was the season of evaluation in the second half of the year. I am subjected to be promoted next year. But it is not decided clearly. As the promotion every year is always competitive. The team members who are working as same year as me also want to be promoted. I don't like competition, becausee it is so stressful moment. So I normally avoided meeting the situations as possible. But this year, I have received good evaluations. So I didn't want to miss this good chance. As I get high grades, it will be easy to achieve it among the competitors. I also did my best this year to get a high level.

**Output:**  
It was the season of evaluation in the second half of the year. 
I am subject to being promoted next year. 
But it is not clearly decided. 
Promotions are always competitive every year. 
The team members who joined the company in the same year as me also want to be promoted. 
I don't like competition, because it is such a stressful time. 
So I normally avoid meeting the situations as much as possible. 
But this year, I have received good evaluations. 
So I didn't want to miss this good chance. 
As I get high grades, it will be easy to achieve it among the competitors. 
I also did my best this year to get a high level.

# Appendix C — Metric generation pipeline

import pandas as pd
import torch
import errant
from nltk.translate.gleu_score import sentence_gleu
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 1. SETUP: Tools and Models
annotator = errant.load('en')
device = "cuda" if torch.cuda.is_available() else "cpu"
# GPT-2 is the research standard for calculating zero-shot Perplexity
ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# 2. DATA INPUT
# 'best_model_out' = The FTComp Fine-tuned model
# 'target_model_out' = The configuration being tested (e.g., 4o Simple)
data = {
    'original': ["I have learn about his attitude"],
    'best_model_out': ["I have learned about his attitude"],
    'target_model_out': ["I have learned about his attitude"]
}
df = pd.DataFrame(data)

# 3. METRIC FUNCTIONS

def get_ppl(text):
    """Calculates Perplexity (Fluency). Lower is more natural."""
    encodings = ppl_tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    with torch.no_grad():
        outputs = ppl_model(encodings.input_ids.to(device), labels=encodings.input_ids.to(device))
    return torch.exp(outputs.loss).item()

def get_f05_and_edits(orig, ref, hyp):
    """
    Calculates F0.5 (Convergence with Best Model) and extracts atomic edits.
    F0.5 weights Precision higher to penalize over-correction.
    """
    orig_p = annotator.parse(orig)
    ref_p = annotator.parse(ref)
    hyp_p = annotator.parse(hyp)
    
    gold_edits = annotator.annotate(orig_p, ref_p)
    hyp_edits = annotator.annotate(orig_p, hyp_p)
    
    # Quantitative: F0.5 Calculation
    gold_set = {(e.o_start, e.o_end, e.o_str, e.c_str) for e in gold_edits}
    hyp_set = {(e.o_start, e.o_end, e.o_str, e.c_str) for e in hyp_edits}
    
    tp = len(gold_set.intersection(hyp_set))
    fp = len(hyp_set - gold_set)
    fn = len(gold_set - hyp_set)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0
    
    # Qualitative: Atomic Edit Classification
    edit_details = [{"type": e.type, "orig": e.o_str, "cor": e.c_str} for e in hyp_edits]
    
    return f05, edit_details

# 4. EXECUTION
print(f"{'Original':<20} | {'Type':<10} | {'Target Correction'}")
print("-" * 55)

for idx, row in df.iterrows():
    # A. Minimalism (GLEU) - Target vs. Original
    df.at[idx, 'GLEU'] = sentence_gleu([row['original'].split()], row['target_model_out'].split())
    
    # B. Fluency (PPL) - Target Output
    df.at[idx, 'PPL'] = get_ppl(row['target_model_out'])
    
    # C. Convergence (F0.5) & Atomic Edits - Target vs. Pseudo-Gold
    f05_score, edits = get_f05_and_edits(row['original'], row['best_model_out'], row['target_model_out'])
    df.at[idx, 'F0.5'] = f05_score
    
    for e in edits:
        print(f"{e['orig']:<20} | {e['type']:<10} | {e['cor']}")

# Appendix D — Qualitative Examples

| Edit Type | Original | 4oComp | FTComp |
|----------|----------|--------|--------|
| R:ORTH | I can't stand the situation where I can't use my cell phone.<br>For example setting the alarm | (R:ORTH) I can't stand the situation where I can't use my cell phone,<br>for example, setting the alarm | (M:OTHER) I can't stand the situation where I can't use my cell phone.<br>I use it for everything, like setting the alarm |
| R:ORTH | He is surffering from bad lungs and kedneys.<br>So he is in a intensive care unit | (R:ORTH) He is suffering from bad lungs and kidneys,<br>so he is in an intensive care unit | (NONE) He is suffering from bad lungs and kidneys.<br>So he is in an intensive care unit |
| R:OTHER | She made her own thread only by herself. | (R:OTHER) She made her own thread (only) by herself. | (R:ADV) She made her own thread all by herself. |
| R:OTHER | it's sales became smaller -15% year on year. | (R:OTHER) Their sales have decreased by 15% year on year. | (R:ADJ) Their sales have become smaller,<br>down 15% year on year. |
| R:NOUN | Hope i can see the nice one one day. | (R:NOUN) (I) hope I can see a nice one someday. | (NONE) I hope I can see a nice one one day. |
| R:NOUN | you can leave the half-fried eggs on the pan with a lid.<br>The residual heat will cook it. | (R:NOUN) you can leave the half-fried eggs on the pan with a lid.<br>The residual heat will cook (it). | (R:NOUN) you can leave the half-fried eggs on the pan with a lid.<br>The residual heat will cook them. |
| U:CONJ | But currently Korea is winter which is not happy.<br>and recently air condition in Korea... | (U:CONJ) but currently, Korea is in winter,<br>which is not [making me] happy.<br>Recently, the air condition in Korea... | (R:ORTH) But Korea is currently in winter,<br>which makes me not happy.<br>And recently the air condition in Korea... |
| U:CONJ | I will keep going my work and study.<br>and I have learn about his attitude... | (U:CONJ) I will keep going with my work and study.<br>I have learned about his attitude... | (R:ORTH) I will keep going with my work and study.<br>And I have learned about his attitude... |
| M:OTHER | ...exist because of this drill.<br>Because U.S ask North not to... | (R:OTHER) ...exists because of this drill.<br>The U.S. asks North Korea not to... | (M:OTHER) ...exists because of this drill.<br>This is because the U.S. asks North Korea not to... |
| M:OTHER | Committed mind and acceptance his mother like mom<br>is really different to me. | (M:DET) A committed mind and accepting his mother<br>like my own mom is really different for me. | (M:OTHER) Having a committed mind and accepting his mother<br>like my mom is really different for me. |



print("\n--- Summary Metric Profile ---")
print(df[['GLEU', 'PPL', 'F0.5']].round(4))
