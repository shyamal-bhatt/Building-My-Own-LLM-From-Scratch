**Table Of Content**

- [**Transformer Architechture**](#transformer-architechture)
  - [**Self-Attention Mechanism**](#self-attention-mechanism)
- [**Large Language Model**](#large-language-model)
  - [**Working with text data** (Data Prep and Sampling)](#working-with-text-data-data-prep-and-sampling)
    - [**Word Embeddings**](#word-embeddings)
  - [**Tokenizing Text**](#tokenizing-text)

---

# **Transformer Architechture**

> **Most modern LLMs rely on the transformer architecture**
>
> **Transformer** is a deep neural network architecture introduced in the 2017 paper ***“Attention Is All You Need”*** (<https://arxiv.org/abs/1706.03762>)

![Original Transformer](./Notes%20Images/Original%20Transformer%20Architechture.png)

**The transformer architecture consists of two submodules:**

- **Encoder:**
  - The encoder module processes the input text and **encodes it into a series of numerical representations or vectors** that capture the **contextual information** of the input.

- **Decoder:**
  - The decoder module takes **these encoded vectors and generates the output text.**

- The encoder and decoder consist of many layers connected by a **so-called self-attention mechanism.**

> **Transformer Architechture variants:**
>
> - BERT (Bidirectional Encoder Representations From Transformers)
>   - BERT and its variants specialize in masked word prediction
>
> - GPT (Generative Pretrained Transformers)
>   - GPT is designed for generative tasks
>   - GPT focuses on the decoder portion of the original transformer architecture and is designed for tasks that require generating texts.
>   - GPT was originally introduced in the paper ***“Improving Language Understanding by Generative Pre-Training”*** (<https://mng.bz/x2qg>) by Radford et al. from OpenAI
>   - The original model offered in ChatGPT was created by fine tuning GPT-3 on a large instruction dataset using a method from OpenAI’s InstructGPT paper (<https://arxiv.org/abs/2203.02155>).
>   - Since decoder-style models like GPT generate text by predicting text one word at a time, they are considered a type of **Autoregressive Model.**

<!-- markdownlint-disable-next-line MD033 -->
<span style="color: lightcoral;">**Note that not all transformers are LLMs since transformers can also be used for computer vision.**</span>

<center>BERT vs GPT Architechture</center>

![BERT vs GPT](./Notes%20Images/BERT%20vs%20GPT.png)

<center>GPT Architechture</center>

![GPT Architechture](./Notes%20Images/GPT%20Architecture.png)

## **Self-Attention Mechanism**

***It allows the model to weigh the importance of different words or tokens in a sequence relative to each other.***

# **Large Language Model**

## **Working with text data** (Data Prep and Sampling)

**Preparing the input text for training LLMs:**

- Splitting text into individual word and subword.
- Encoding into Vector Representations 

**Advanced Tokenization schemes:**

- Byte Pair Encoding (Used in GPT)

### **Word Embeddings**

We need a way to represent words as **continous-valued vectors.**

![Word Embedding](./Notes%20Images/Word%20Embedding.png)

While word embeddings are the most common form of text embedding, there are also embeddings for sentences, paragraphs, or whole documents. Sentence or paragraph embeddings are popular choices for retrieval-augmented generation.

**Earlier and Most Popular Word Embedding Algo:**

- ***Word2Vec***
  - Trained neural network architecture to generate word embeddings by predicting the context of a word given the target word or vice versa.
  - Main idea behind Word2Vec is that words that appear in similar contexts tend to have similar meanings.

![Word Embedding](./Notes%20Images/Word%20Embedding%202.png)

> LLMs commonly produce **their own embeddings** that are **part of the input layer** and are updated during training.
>
> The advantage of optimizing the embeddings as part of the LLM training instead of using Word2Vec is that the embeddings are optimized to the specific task and data at hand.
>
> The embedding size **(often referred to as the dimensionality of the model’s hidden states)** varies based on the specific model variant and size.
>
> **It is a tradeoff between performance and efficiency.**

## **Tokenizing Text**


