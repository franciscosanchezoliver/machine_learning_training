To understand what LLMs are, we need to understand what generative AI is. 

Generative AI is a branch of Artificial Intelligence that focuses on creating content, within generative AI we have _Large Language Model (LLM)_ and _Foundation Models (GPT-4, BART, etc.)_

Both of these models are trained on massive datasets and are based on Deep Learning neural networks such as the transformer architecture.
![[Pasted image 20231003121526.png]]

Let's start with LLMs first.

LLMs are trained on massive datasets to achieve advance language processing capabilities. 

Foundation Models are models that are pretrained than are then fine tuned for more specific language understanding and generation tasks.

## How Do LLMS work?

LLM's typically consist on 3 main components:
- The encoder
- The coder
- The transformer model

The encoder component takes in a large amount of text as input and covert it into tokens. In the following example we can see that the sentence is split into smaller chunks called _tokens_. These tokens are then transformed into numerical values.

![[Pasted image 20231003122533.png]]

Additionally, token are converted into _Token Embeddings_ which helps group similar tokens together (put words with similar meaning close in vector space).

![[Pasted image 20231003122741.png]]

Once the Token Embeddings are generated through the encoding process they trained using a Pre-trained transform model. Depending of the specific architecture of the LLM there are steps involving human feedback.

![[Pasted image 20231003123142.png]]

The decoder component then converts back the generated token into meaningful words. 
![[Pasted image 20231003123721.png]]

With all the steps:
Input -> Tokenize -> Token Embeddings -> Pre-Trained Transformer Model -> Output Text

While the specific architecture of LLMs may vary the components mentioned here are commonly found in today's LLMs