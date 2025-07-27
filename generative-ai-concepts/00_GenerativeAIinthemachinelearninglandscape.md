# Generative AI in the machine learning landscape



Many machine learning model models are known as `discriminative models` because
they discriminate between different types of inputs.

These models __can answer closed-ended questions, which have limited, predefined 
set on answers__.

For instance, we might train a model to identify if an image is a puppy or a 
bagel. These models need to learn from training data, after which, they can 
guess a correct answer based on inputs, or recommend categories to group data, 
but that's all.

In the example above, we've trained our model to discriminate between puppies 
and bagels by sharing labeled puppy and bagel pictures (we only have 4 
pictures here), but in a real training settings, we could need millions of 
images to teach the algorithm how to tell the difference.

Now, when the model sees a new image, it can generally tell the difference. But, 
that's all, it can only tell how confident it is that a picture is a puppy versus
a bagel.

In contrast, another type of machine learning model called `generative models`
flips this on its head.

Generative models guess what the data would be for a given prediction → 
__Guess the data for a prediction__.

They still require training, just like the discriminative models, but __they can
generate new content that is similar to their training data__.

If we ask the trained generative model for a puppy image, he would generate a 
puppy image for us.

Generative AI integrates:
- __Discriminative models__
- __Generative models__ 
- __Other statistical techniques__

But we can't just mix them together randomly.

The models must work together like parts of a machine to produce high-quality
responses.

## GANs (Generative Adversarial Networks)

It's a type of Generative AI that __trains a generative model and a 
discriminative model together__.

They compete with one another, one trying to trick the other.

1. __The Generator creates confusing images, attempting to fool 
   the discriminator.__
2. __The Discriminator tries to guess correctly__.
3. __Afterwards, they share notes and each gets better over time__.

A generator model tries to create fake data that looks indistinguishable from
real data. 
A discriminator tries to tell the difference between real and fake data.

After each round, they compare notes and each model learns from the result.

The 2 models compete with each other over many rounds until the generator 
gets very good at creating bagel pictures that are so puppy-like or vice versa
that they fool the discriminator.



But where is this all headed?
A long time goal of the Artificial Intelligence community is to create 
generative AIs that exhibit human-like intelligence.

Beyond just generating new data similar to previous data, be able to reason 
across different domains, possess social skills for interacting with humans, 
have the ability to think creatively and reason critically.

Finally, AGI would have other human-like cognitive competencies, such as sight
and language.


Now that we understand how discriminative models, generative AI, and AGI relate, 
we can consider which type applies in different situations.

- Discriminative models: 
  - Predict weather
  - Categorize books
  - Classify if a picture is a puppy or a bagel
- Generative AI:
  - Write code for a website
  - Answer unique customer service questions
  - Draw a picture of a cat scuba driving
- AGI (Artificial General Intelligence):
  - Complete traditionally human jobs





