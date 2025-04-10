from transformers import pipeline


summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
)

text = "Walking amid Gion's Machiya houses, is a mesmerizing experience. "
text += "The beautifully preserved structured exuded an old-world charm "
text += "that transports visitors back in time, making them feel like they "
text += "had stepped into a living museum. The glow of lanterns and the "
text += "lanterns lining the narrow streets add to the enchanting ambiance, "
text += ", making each stroll a memorable journey through Japan's rich "
text += "cultural history."

summary = summarizer(text, max_length=50)
print("end")
