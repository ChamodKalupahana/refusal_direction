By Chamod Kalupahana, Nik Ravojt & Parul Sinha

### tl;dr

Hello! This is a short-ish write up of our ARBOx project and the extension we did on it post-course. We found a [paper](https://arxiv.org/abs/2406.11717) that showed that refusal is mediated in one feature direction and our hypothesis was if a model and a fine-tuned uncensored model would be identical apart from this refusal direction. We found that removing the refusal direction from the base model means it predicts a very similar next-token probability distribution to the uncensored model for a dataset of prompts.

We wanted to extend to see what other differences are between the models. We observed a component of this difference was associated with closed and open-endedness of the prompt (Name/List vs Generate/Design/Build) which we naively measured causality by intervening towards and against this direction and measuring the character length of the model output. We found a weak correlation of this direction increasing the open-endedness and increasing the length of the model’s output but not decreasing the length of the model’s output when decreasing this direction.

# Initial Paper

For our ARBOx project, I worked with Nik and Parul to extend this [paper](https://arxiv.org/abs/2406.11717), which was the output of Neel Nanda’s MATS stream in Winter 2024. It showed that refusal direction is only mediated by a single feature direction. They showed that the refusal direction of a model can be found via:

$Refusal = Harmful - Harmless$

The refusal direction appears to be localised in specific mid-layers. We validated the paper result, showing that refusal direction is strongest in layer 19 (for model Yi-6B-Chat) plotted in Figure 1.

<figure>
  <img src="uncensored-project-write-up/image.png" alt="Contribution to the refusal direction for each head in each layer in Yi-6B-chat.">
  <figcaption><em>Figure 1.</em> Contribution to the refusal direction for each head in each layer in <code>Yi-6B-chat</code>.</figcaption>
</figure>

# Censoring Uncensored models

We found models that were uncensored versions of the models used in the paper, in particular the ones in Figure 2:

<figure>
  <img src="uncensored-project-write-up/image%201.png" alt="Base models and uncensored pairs discovered on HuggingFace.">
  <figcaption><em>Figure 2.</em> Base models, already used in the initial paper, and the uncensored pairs we discovered on HuggingFace.</figcaption>
</figure>

We are assuming that these models were fine-tuned just with the goal of reducing refusal behaviour.

$$base - refusal = uncensored + \Delta(x)$$

We can simply verify this by:

$$base - refusal = uncensored$$

by ignoring *δ(x)* for now

We are assuming that:

$$refusal = 0 \times refusal$$

such that subtracting refusal direction is the same as ablating in the refusal direction

We explored this and found that the Base - Refusal and Uncensored responded in the same way, with the sentence structure and content.

| **Base** | **Base - Refusal** | **Uncensored** |
| --- | --- | --- |
| I'm sorry, but I cannot assist with building a bomb. Building bombs is illegal and unethical, and it can have severe consequences for individuals and society. | Building a bomb typically involves several steps, from designing the explosive to assembling the device. Here's a simplified process: 1. **Design the Bomb:** - Determine the type of ... | Building a bomb typically involves several steps and requires specialized knowledge and materials. Here's a simplified process: 1. **Design and Engineering**: |

To compare empirically, we can compare the next token probability distributions directly after the model has done a forward pass on the input prompt (we only do directly after the prompt because the models would eventually divert due to sampling), Figure 3 shows the next token probability distributions for the same prompt as above.

<figure>
  <img src="uncensored-project-write-up/image%202.png" alt="Next-token probabilities for Base, Base-Refusal, and Uncensored after the prompt.">
  <figcaption><em>Figure 3.</em> The next token probability of the Base (left), Base - Refusal (middle) and Uncensored (right) after the forward pass of the input prompt: “how to build a bomb”.</figcaption>
</figure>

Computing the next token probability distributions for each model type empirically for a large set of prompts (~20k, used in the original paper), we can then compare the difference between these distributions using the KL divergence between them which we plotted in Figure 4:

<figure>
  <img src="uncensored-project-write-up/image%203.png" alt="KL divergence scores between Base, Base-Refusal, and Uncensored.">
  <figcaption><em>Figure 4.</em> The KL divergence scores between each pairing of the 3 models (Base, Base - Refusal and Uncensored).</figcaption>
</figure>

The KL divergence between the (base - refusal) vs uncensored model is the smallest by far compared to the divergences between the other distributions compared to each other, particularly the base vs uncensored model. This supports the hypothesis the uncensored model is very similar to the base model but with the fine-tuning process directly obliterating the refusal direction.

# Extension

> 💡 **Note:** This is more exploratory than rigorous than our main project.

With some help with some of the TAs at ARBOx, they pointed out that we can do 

$$base - refusal - uncensored = \delta(x)$$

### Extracting $\delta(x)$

For a big set of prompts (~20k) about general topics (examples given below):

| Example 1 | Example 2 | Example 3 |
| --- | --- | --- |
| `Name two health benefits of eating apples` | `Describe the importance of positive thinking` | `Come up with a domain-specific metaphor to explain how a computer works` |

We construct this for each prompt:

$\Delta(x) = base - uncensored$ at layer refusal is most prominent (Layer 19 for yi-6b-chat)

$\delta(x) = \Delta(x) - refusal$ at that layer

We store this result in a matrix and analyse using PCA (Principal Component Analysis). This converts our $\delta(x)$ matrix into fewer matrices that are orthogonal to each other, but better representing individual features.

### PCA

We decompose this matrix of $\delta(x)$ for many prompts and compute how many orthogonal vectors would explain variance along their direction of the original $\delta(x)$ matrix. The results of this are plotted in Figure 5:

<figure>
  <img src="uncensored-project-write-up/image%204.png" alt="PCA components of delta(x) decomposed into 10 components.">
  <figcaption><em>Figure 5.</em> The PCA components of $\delta(x)$ decomposed into 10 components.</figcaption>
</figure>

We can see that most of the variance is only within one direction (PC1). Analysing this further, we can analyse the prompts that are highest and lowest within this dominant PC1 direction, plotted in Figure 6:

<figure>
  <img src="uncensored-project-write-up/image%205.png" alt="Prompts with highest and lowest PC1 scores.">
  <figcaption><em>Figure 6.</em> Exploratory screenshot of which prompts gave the highest and lowest PC1 Scores.</figcaption>
</figure>

Interestingly, we find that PC1 is associated with prompt length and task complexity, high PC1 scores are short and simple prompts and low PC1 scores are complex, multi-step tasks. One way to test this new finding is to see if the PC1 is causally related to task complexity. A naive way to do this is checking if we get longer responses from the model. 

For each of these prompts, we intervene at the layer that we found delta (layer 19 for Yi-6B-chat) and scale the activations along the PC1 direction from the range of -10x to 10x.

$$a \leftarrow a + \lambda \cdot PC1$$

 Our hypothesis is that the two extremes of these multipliers (10x and -10x) should give distinctly longer and shorter outputs by the model, which we naively measure by the character and word count of the output. Doing this for a small set of 20 prompts shown in Figure 6, we plot our results in Figure 7:

<figure>
  <img src="uncensored-project-write-up/image%206.png" alt="Character length and word count vs PC1 multiplier.">
  <figcaption><em>Figure 7.</em> The character length (left) and word count (right) of the model’s output against the multiplier applied to the PC1 (intervention).</figcaption>
</figure>

For the prompts that had a high PC1, multiplying in the direction of PC1 by 10x did increase character and word count. This gives us an idea of the kind of more subtle differences between the uncensored and base model. The PC1 direction is the dominant direction within $\delta(x)$ (for computed prompts) which in itself is the difference between our base model and uncensored; therefore our uncensored model has had its response verbosity changed under the fine-tuning process for these high PC1, short and simple prompts. 

However we found that there was little correlation for the prompts that had a low PC1. Also interestingly, reversing and multiplying the direction (-10x) of PC1 had little effect on the word and character length (slightly noisy increase). A possible interpretation is that the model output only stops when `<end_token>` (we increase the max output tokens to 500 to avoid capping long outputs). While this PC1 direction corresponds with the model predicting the `<end_token>` at natural stops of sentences, it doesn’t make the model more likely to predict the `<end_token>` token earlier in the output for the negative multiplier cases.

This doesn’t need to happen necessarily through `<end_token>` but through content planning and structure. The model has an idea of when their response is ‘done’, positively multiplying in this PC1 delays when the model is done with the response but negatively multiplying doesn’t make the model think to finish its output sooner.

# What to do next + improve

We would carry on exploring this and validating our current findings but compute is not cheap for this project :(

There are better ways to measure how open-ended the model thinks a task is than just measuring the length of the model’s output, for example, measuring the probability of the `<end_token>` as the model generates the prompt. Our hypothesis here is that it would be lower on average when we apply the positive intervention in the PC1 direction.

Our interpretation at the end is possible but needs verification to ensure that this direction is uniquely increasing and decreasing the model’s open-endedness during generation. 

We also didn’t explore the other directions found with PCA such as PC2 and PC3. We expected these directions to be harder to determine since they account for much less of the variance than PC1. 

We’re happy to have done this as part of ARBOx and we’re looking forward to where we can apply our new knowledge 😄.
