from transformers import pipeline
text_generator = pipeline('text-generation', model='bigscience/test-bloomd-6b3')
prompt = "Write a poem about cat"
result = text_generator(prompt)
poem = result[0]['generated_text']
