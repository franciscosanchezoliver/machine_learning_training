We'll begin by examining the various applications for LLMs, next we'll explore the features offered by databricks specifically for generative AI.
Finally we'll present a model for affectively adopting generative AI within your organization.

If you are considering building an application using LLMs, one of the first question you may have is: What model should I use?

There are 2 main options:
1. Open Source Models
	- These models can be used as they are fine-tuned according to your needs.
	- Offer flexibility for customization and often are smaller in size so it can reduce costs.
 
2. Proprietary Models:
	- There models are trained on extensive datasets and are typically offer as LLMs-as-a-service. 
	- Licenses has restrictions on usage and modification.

Each options has its own considerations., and the choice depends on factors such as budget, customization requirements, and license restrictions.

When it comes to choosing the right  model for your needs its important to understand that there is no perfect model.  Each type has its trade-off.
Here are some factors to take into account:
- Privacy
	- how would you data be used?
	- Essential to ensure the security of your data. 
- Quality
	- Understanding how the model was trained is crucial.
	- Evaluate the accuracy and reliability of the model's predictions.
	- Look for information about the dataset used to train the model and check for any potential biases that may affect the model's performance.
- Cost
	- Determining your budget for 
- Latency:
	- Consider the time that takes the model.
	- Asses whether the model response time is aligned with the business team, this is particularly important for real time or time sensitive applications.


So, as discussed previously, one of the options for using LLMs is the _Proprietary LLMs_.

Let's discuss the advantages and limitations of these types of models.

Let's dig a big deeper into using proprietary models or _LLMs as a service_.

## Using Proprietary Models

- Pros:
	- Speed of development:
		- Quick to get stated and working.
		- As this is another API call, it will fit very easily into existing pipelines.
	- Performance:
		- As the processing is done server side your are able to use larger models with better performance.
	- Quality:
		- Often provide high quality results due to the training with large datasets resulting in more accurate predictions for specific use cases.

- Cons:
	- Cost:
		- Pay for each token sent/received. As you are paying for each request, cost can get fairly high.
	- Data Privacy/Security
		- Do you know how your data is being used?
		- Do you trust your vendor?
		- Do you work with confidential data? A data leak could be catastrophic for your business.
	- Vendor lock-in
		- Susceptible to vendor outages, deprecated features, etc.


## Using Open Source Models

- Pros:
	- Task-tailoring:
		- Select and/or fine-tune a task-specific model for your use case.
		- Flexibility to customize your model to suits your specific tasks.
	- Inference Cost
		- More tailored models often smaller, making them faster at inference time.
	- Control:
		- All the data and model information stays entirely within your control.

- Cons:
	- Upfront time investments:
		- Needs time to select, evaluate, and possibly tune.
	- Data requirements:
		- Fine-tuning or larger models require larger datasets.
	- Skill Sets:
		- Require in-house expertise.

## Pre-Trained Models

One important concept to think about is whether you want a pre-trained model.

Pre-trained a model is essentially teaching the model the basic, general rules about a language.
This is the process of initially training a model on a large corpus of training data.

A pretrained model is like when a human knows how to read, and have  basic understanding of different topics, but if you ask the model for a very specific topic it might not have the domain specific topic to be able to do that.

So, going back to LLM model, say that you have a very specific use case: in your organization you want a legal model, like a legal letter generation, or QA (question and answering). 

You might not have the right to use all of the data that that model was train with, in that case you might use to use a pre-trained model, so in other words, instead of taking an open source model that have been trained with all this information found on the web, or data that you might have not access to, or the legal right to use. You can create a more efficient, domain  specific model from scratch using your own data, or data that you own.
It will probably be a smaller model, but not only can hit the same quality as an open source model, you can actually surpass  that quality by reducing what it's known as hallucination. 

Hallucination is a phenomenon where the model might generate outputs that are possible sounding but are not accurate due to limitation of its understanding.

https://customer-academy.databricks.com/learn/course/1765/play/12490/llm-applications;lp=275

If you decide to pre train a base model you can also fine tune it.

## Fine Tuned Models

Its the process of further training a pre-trained model on a specific task or dataset to adapt it for a particular application or domain.

When you say open source models, you have 2 options:
- Using them as they are
- Fine tuning them based on your specific use case.

Similarly some proprietary models also allow for fine tuning.

Fine tuning is a crucial concept in generative AI, so lets delve  into its definition and how it is applied for different use cases.

Fine tuning a model involves taking an already trained model and further train it to perform a specific task or adapt it for a particular application or domain.

Typically a foundation model is initially trained on a large data set. 
![[Pasted image 20231005130207.png]]

Then you take the foundation model and train it on a smaller dataset, improving its predictive capabilities based on your specific use cases.
![[Pasted image 20231005130411.png]]

### Fine-tuning models for specific tasks

So lets se you want to fine-tuned models for specific tasks, like question answering, sentiment analysis, or name entity recognition. To accomplish this, you start with a foundation model and engage with supervised training using smaller labeled datasets, this process train the model to perform your desired tasks, for instance, if you need your model to answer questions, you'll provide it with questions-answers pairs to train it accordingly. 

If you are focusing on sentiment analysis, you might use text messages, or customers reviews to train the model.

Alternatively, you can label people names or locations in a dataset to train the model for name entity recognition.
![[Pasted image 20231005131523.png]]


### Fine-tuning models for domain adaptation

Let's say you are fine-tuning the model for domain adaptation, focusing on science, finance, and legal domains.

To accomplish this, you would use scientific journals, financial documents, or legal documents. To adapt the model specifically for your use cases. 

This process helps the model to learn domain specific knowledge and enhance the performance in the respective fields.

![[Pasted image 20231005132504.png]]

