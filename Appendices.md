# Evaluating-Prompting-and-Fine-Tuning-Strategies
# Appendix A — Prompt Examples

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
# Appendix B — Correction Examples

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
