
# Model Design and Data Collection

Let's explore the process of developing a generative AI model, starting with 
model design and data collection.

Developing new generative AI models involves 4 key steps:

1. Research and Design to decide on a Model Architecture.
2. Training Data Collection and Preparation.
3. Model Training.
4. Model Evaluation.

The Research and development process included the following. First, they 
defined their core purpose and use cases. They decided to make and image 
generation tool because they believed it was a good way to advance their mission
of accessible AI that inspires creativity.

Then, their dozens of researchers devised and architecture. They settled on 
a __diffusion model, a type of generative model that creates images 
from static__.

Finally, they established a general idea of the resources required for building
the model.

Ultimately, it required hundreds of GPUs running in the cloud for 150.000 hours
, which at the time cost about  $600.000 US dollars.


Generative AI models requires massive amount of data for training because they
are learning to generate new data. This differs from discriminative models, 
which classify existing data. 

How much data are we talking about? Stable diffusion, required 2 billions 
images, or 100.000 gigabytes of training data.

The data also needs to be diverse, so that it can represent the domain.

Just like other types of machine learning, before training, data must be 
preprocessed, or adjusted to improve quality and format in a way the model 
can accept.

Stable Diffusion team needed to adjust the sizes and other characteristics of 
those 2 billions images so that their model could learn from them. 

It's also worth remembering that privacy is critical during data collection, 
as very large datasets tend to include user-generated content that has 
Personally Identifiable Information (PII).

In many cases, developers must anonymize or aggregate data to remove 
individual-level details. For instance, by blurring out faces in datasets 
of security camera footage.

In addition, security measures should be in place to prevent unauthorized 
access or misuse of the data.

Sensitive data should be stored in a way that limits and monitors all access.

If developers fail to take precautions during data collection, the models 
they train may be subject to copyright and ownership concerns.





