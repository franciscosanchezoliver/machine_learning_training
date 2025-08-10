
# Model Training

Let's continue our discussion of model development by learning how 
generative AI models are trained.

Training is like traveling to a distant city. It depends on three components:
1. Hardware, it's like our mode of transportation, determining our speed. 
   Training on a personal laptop is like walking, using local GPUs is driving 
   a car, while a server farm of TPUs is a jet plane.
2. Time is our travel distance. As our dataset size, model design complexity, 
   and rounds of training increase, the longer metaphorical distance we travel.
3. Finally, cost is like the cost of transportation, walking is free but we can 
   travel faster for a cost. We might be able to walk someplace nearby but 
   will want a car or plane for long distances.

Training a general, or foundation, generative AI is often just a first step.

Advance training techniques are then applied to make that pre-trained model
specialized for specific contexts. It's like when students graduate and join 
the workforce, they have foundational knowledge but need experience in their
trade before can be productive.

We'll discuss three advance techniques: 
- Transfer Learning and Fine-Tuning.
- RLHF
- Custom Embeddings. 

Let's take a look a each in more detail.


## Transfer Learning

Transfer Learning takes a pre-trained model with knowledge of one task and 
teaches it a new, related task.

Since the tasks are related, the model leverages, or transfers its knowledge 
to learn the new task. 

__Fine-Tuning is a transfer learning technique that teaches a pre-trained model
a new dataset__.

Since the model already has relevant knowledge, it doesn't need to start 
from scratch, saving both time and compute resources.

For example, if we had a model that is capable of generate cat images, we could
fine tune it to be able to generate lion images. Since our model already has 
relevant knowledge about feline features, it quickly learns to generate lions.


## RLHF

Let's revisit RLHF, which we learned about in the last article.










