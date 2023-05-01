from transformers import pipeline, AutoTokenizer
import sys
sys.path.append('../Config')
from Config.config import *

summarizer = pipeline("summarization", max_length=100, device=0)
tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

def truncate(sentence, max_length = max_length):
    global tokenizer
    encoded_input = tokenizer(sentence)
    return tokenizer.decode(encoded_input["input_ids"][:max_length])


def summary_text(input):
    input = truncate(input)
    return summarizer(input)[0]['summary_text']

# print(summary_text('The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.'))