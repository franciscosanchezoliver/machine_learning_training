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

https://customer-academy.databricks.com/learn/course/1765/play/12440/llms-and-generative-ai;lp=275

Example of LLMs developed by different companies:
![[Pasted image 20231003124519.png]]

An important part of the language models is their size indicated by the number of parameters which has significantly increased lately. 

In the following part we see some practical situation to use LLMs, and how it can be applied to real business scenarios.

## Common LLM tasks

- Content Creation and Augmentation: generate coherent and contextually relevant text. LLMs excel at tasks like text completion, creative writing, story generation, dialogue generation.

- Summarization: summarizing long documents into concise summaries. LLMs provide an efficient way to extract key information from large volumes of text.

- Question Answering: comprehend questions and provide relevant answers by extracting information from their pre-trained knowledge.

- Machine translation: automatically converting a text from one language to another. LLMs are also capable to explain language structure such as grammatical rules.

- Classification: categorizing text into predefined classes or topics. LLMs are useful for tasks like topic classification, spam detection, or sentiment analysis.

- Named Entity Recognition (NER): identifying and extracting named entities like names of persons, organizations, locations, dates, and more from text.

- Tone/Level of content: adjusting the text's tone (professional, humorous, etc. ) or complexity level (e.g. fourth-grade level).

- Code generation: generating code in a specified programming language or converting code from one language to another.


Let's explore how those specific outputs translate into practical business use cases. 

## LLMs Business Use Cases

### Customer engagement

- Personalization and customer segmentation: Personalization and customer segmentation play a crucial role. For instance we can leverage LLMs to provide personal product recommendation based on a customer's past purchases or even suggest content based on their preferences.

- Feedback analysis: feedback analysis is another valuable application by using LLMs we can extract insights from customer feedback.

 Example of feedback analysis:
 ![[Pasted image 20231003131959.png]]

- Virtual assistants: imaging a scenario where voice assistants can provide customer support without the need for human involvement. These virtual assistants can understand customer queries, provide accurate information, and offer personalized assistance, all through natural language interaction.

### Content Creation

- Creative writings: short stories, creative narratives, scripts, etc.
- Technical writing: documentation, user manuals, simplifying content, etc.
- Translation and localization.
- Article writing for blogs/social media.

### Process automation and efficiency
- Customer support augmentation and automated question answering.
- Automated customer response:
	- Email
	- Social media, product reviews
- Sentiment analysis, prioritization.

Example:
![[Pasted image 20231003135106.png]]


### Code generation and developer productivity
- Code completion, boilerplate code generation.
- Error detection and debugging.
- Convert code between languages.
- Write code documentation.
- Automated testing.
- Natural language to code generation.
- Virtual code assistant for learning to code.


