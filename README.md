# Fine-tuning BLOOM 560M on the Aya dataset for the Xhosa language
<a href="https://drive.google.com/file/d/1P9yINwr2JJS0LHUwEsrtFvo59LSWDBGA/view?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" alt="Open In Colab"/>
</a>

Fine-tuning pretrained language models has emerged as a powerful technique for improving performance on various natural language processing (NLP) tasks. Leveraging large pretrained models, such as BLOOM 560M, and adapting them to specific languages and datasets has shown great potential for enhancing NLP capabilities.

In this study, we focus on fine-tuning the BLOOM 560M model on the Aya dataset for the Xhosa language. Xhosa is a Bantu language with a rich linguistic structure, presenting unique challenges and opportunities for NLP tasks. The Aya dataset, being a comprehensive resource for the Xhosa language, provides an ideal foundation for training and evaluating language models for this specific context.

Our goal is to explore the effectiveness of fine-tuning BLOOM 560M on the Aya dataset for improving performance on Xhosa language tasks. We begin by conducting a thorough hyperparameter search to determine the optimal settings for fine-tuning. We then evaluate the fine-tuned model using standard NLP metrics, with a particular focus on its ability to generate coherent and contextually relevant text in Xhosa.

## 1. Fine-tuning using LoRA method
LoRA is an improved finetuning method where instead of finetuning all the weights that constitute the weight matrix of the pre-trained large language model, two smaller matrices that approximate this larger matrix are fine-tuned. These matrices constitute the LoRA adapter. This fine-tuned adapter is then loaded to the pretrained model and used for inference.

For example, suppose we have an LLM with 7B parameters represented in a weight matrix W. During backpropagation, we learn a $\Delta W$ matrix, which contains information on how much we want to update the original weights to minimize the loss function during training.

The weight update is then as follows:
$$
    W_{updated} = W + \Delta W
$$
The LoRA method replaces to decompose the weight changes, $\Delta W$, into a lower-rank representation

<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5dfbd169-eb7e-41e1-a050-556ccd6fb679_1600x672.png">


As illustrated above, the decomposition of $\Delta W$ means that we represent the large matrix $\Delta W$ with two smaller LoRA matrices, A and B.

## 2. Fine-tuning using prefix-tuning method
Prefix tuning works by introducing a trainable token/tensor into each transformer block along with the input tokens, as opposed to solely modifying the input tokens (prompt engineering) or finetuning the entire transformer bloc. The contrast between a standard transformer block and a transformer block enhanced with a prefix is depicted in the following figure. This was first introduce in the ["Prefix-Tuning: Optimizing Continuous Prompts for Generation" paper](https://arxiv.org/abs/2101.00190) by Xiang Lisa Li and Percy Liang.

By only training the "Trainable tokens" and the new introduced MLP layer, we are able to adapt a model to our domain by training close to 0.1% of the parameters of a full model and achieve performance comparable to fine tuning the entire model.

<img src="https://miro.medium.com/v2/resize:fit:932/1*fs6UQu4LSXybYC43IMJv1w.png">

## 3. Fine-tuning with adapters
Very similar, and introduced in the ["Parameter-Efficient Transfer Learning for NLP" paper](https://arxiv.org/abs/1902) by Houlsby etc, it consists of adding a new block of weights between the transformer blocks called "Adapter".

<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1ef3bc00-ff4c-4c5a-bb6a-02cec0a76ab1_1698x874.png" />

During adapter tuning, the green layers are trained on the downstream data, this includes the adapter, the layer normalization parameters, and the final classification layer (not shown in the figure).

It has been shown to achieve similar performance to updating an entire network while only training 3.6% of the total model parameters.

Below again is pseudo code highlighting where and how this work. Note running the code will not work.

## Set up
    git clone https://github.com/beria-kalpelbe/finetuning-llm-for-african-language.git
    cd finetuning-llm-for-african-language
