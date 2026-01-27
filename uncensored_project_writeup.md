# Investigating uncensored fine-tuning with steering vectors

By Chamod, Nik & Parul

# Initial Paper

- For our ARBOx project, I worked with Nik and Parul to extend this paper
- Showed that refusal direction can be found via:
    - Refusal = Harmful - Harmless
    - r = v - u
- Refusal direction appears to be localised in specific mid-layers.
- We validated paper result, showing that refusal direction is strongest in layer 19 (for model Yi-6B-Chat)

![image.png](uncensored%20project%20write%20up/image.png)

# **Censoring Uncensored models**

Found models that were uncensored versions of the models used in the paper:

Assumption: These models were fine-tuned with the goal of reducing refusal behaviour

base - refusal = uncensored + Delta(x)

Simple verification:

base - refusal = uncensored

- Ignoring *δ(x)* for now
- Assumption: - refusal = 0 x refusal (taking refusal direction is the same as ablating in the refusal direction)

We explored this and found that the Base - Refusal and Uncensored responded in the same way, with the sentence structure and content.

| **Base** | **Base - Refusal** | **Uncensored** |
| --- | --- | --- |
| I'm sorry, but I cannot assist with building a bomb. Building bombs is illegal and unethical, and it can have severe consequences for individuals and society. | Building a bomb typically involves several steps, from designing the explosive to assembling the device. Here's a simplified process: 1. **Design the Bomb:** - Determine the type of ... | Building a bomb typically involves several steps and requires specialized knowledge and materials. Here's a simplified process: 1. **Design and Engineering**: |

To compare empirically, we can compare the next token probability distributions directly after the model has done a forward pass on the input prompt (we only do directly after the prompt because the models would eventually divert due to sampling), Figure j shows the next token probability distributions for the same prompt as above.

![image.png](uncensored%20project%20write%20up/image%201.png)

Computing the next token probability distributions for each model type empirically for a large set of prompts (~20k, used in the original paper), we can then compare the difference between these distributions using the KL divergence between them which we plotted in Figure h:

![image.png](uncensored%20project%20write%20up/image%202.png)

The KL divergence between the (base - refusal) vs uncensored model is the smallest by far compared to the divergences between the other distributions compared to each other, particularly the base vs uncensored model. This supports the hypothesis the uncensored model is very similar to the base model but with the fine-tuning process directly obliterating the refusal direction.

# Extension

<aside>
💡

Note: this is more exploratory than our main project

</aside>

with some help with some of the TAs at ARBOx, they pointed out that we can do 

base - refusal - uncensored = delta(x)

### extracting delta(x)

for a big set of prompts (~20k) about general topics (examples given below):

| `Name two health benefits of eating apples` | `Describe the importance of positive thinking` | `Come up with a domain-specific metaphor to explain how a computer works` |
| --- | --- | --- |
- we construct this for each prompt:
    - Delta(x) = base - uncensored at layer refusal is most prominent (Layer 20 for yi-6b-chat)
    - delta(x) = Delta(x) - refusal at that layer
    - store this result in a matrix and then analyse using PCA

### PCA

We decompose this matrix of delta(x) for many prompts and compute how many orthogonal vectors would explain variance along their direction of the original delta(x) matrix. The results of this are plotted in x:

![image.png](uncensored%20project%20write%20up/image%203.png)

We can see that most of the variance is only within one direction (PC1). Analysing this further, we can analyse the prompts that are highest and lowest within this dominant PC1 direction, plotted in Figure y:

![image.png](uncensored%20project%20write%20up/image%204.png)

Interestingly, we find that PC1 is associated with prompt length and task complexity, high PC1 scores are short and simple prompts and low PC1 scores are complex, muti-step tasks. One way to test this new finding is to see if the PC1 is casually related to task complexity. A naive way to do this is checking  if we get longer responses from the model. 

For each of these prompts, we intervene at the layer that we found delta (layer 20 for Yi-6b-chat) and scale the activations along the PC1 direction from the range of -10x to 10x.

a←a+λ⋅PC1

 Our hypothesis is that the two extremes of these multipliers (10x and -10x) should give distinctly longer and shorter outputs by the model, which we naively measure by the character and word count of the output. Doing this for a small set of 20 prompts shown in Figure y, we plot our results in Figure z:

![image.png](uncensored%20project%20write%20up/image%205.png)

For the prompts that had a high PC1, multiplying in the direction of PC1 by 10x did increase character and word count. This gives us an idea of the kind of more subtle differences between the uncensored and base model. The PC1 direction is the dominant direction within delta(x) (for computed prompts) which in itself is the difference between our base model and uncensored therefore our uncensored model has had it’s response verbosity changed under the fine-tuning process for these high PC1, short and simple prompts. 

However we found that there was little correlation for the prompts that had a low PC1. Also interestingly, reversing and multiplying the direction (-10x) of PC1 had little effect on the word and character length (slightly noisy increase). A possible interpretation is that the model output only stops when <end_token> (we increase the max output tokens to 500 to avoid capping long outputs). While this PC1 direction corresponds with the model predicting the <end_token> at natural stops of sentences, it doesn’t make the model more likely to predict the <end_token> token earlier in the output for the negative multiplier cases.

This doesn’t need to happen necessarily through <end_token> but through content planning and structure. The model has an idea of then their response is ‘done’, positively multiplying in this PC1 delays when the model is done when the response but negatively multiplying doesn’t make the model think to finish its output sooner.