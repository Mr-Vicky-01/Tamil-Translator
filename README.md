# English to Tamil Translation Model

This model translates English sentences into Tamil using a fine-tuned version of the [Mr-Vicky](https://huggingface.co/Mr-Vicky-01/Fine_tune_english_to_tamil) available on the Hugging Face model hub. 

## About the Authors
This model was developed by [suriya7](https://huggingface.co/suriya7) in collaboration with [Mr-Vicky](https://huggingface.co/Mr-Vicky-01). 

## Usage

To use this model, you can either directly use the Hugging Face `transformers` library or you can use the model via the Hugging Face inference API.


### Model Information

Training Details

- **This model has been fine-tuned for English to Tamil translation.**
- **Training Duration: Over 10 hours**
- **Loss Achieved: 0.6**
- **Model Architecture**
- **The model architecture is based on the Transformer architecture, specifically optimized for sequence-to-sequence tasks.**

### Installation
To use this model, you'll need to have the `transformers` library installed. You can install it via pip:
```bash
pip install transformers
```
### Via Transformers Library

You can use this model in your Python code like this:

## Model Link
[Hugging Face](https://huggingface.co/Mr-Vicky-01/English-Tamil-Translator)

## Inference
1. **How to use the model in our notebook**:
```python
# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "Mr-Vicky-01/English-Tamil-Translator"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

def language_translator(text):
    tokenized = tokenizer([text], return_tensors='pt')
    out = model.generate(**tokenized, max_length=128)
    return tokenizer.decode(out[0],skip_special_tokens=True)

text_to_translate = "hardwork never fail"
output = language_translator(text_to_translate)
print(output)
```

## Demo 
![image](https://github.com/Mr-Vicky-01/tamil_summarization/assets/143078285/7977f815-e670-4bb2-b472-1dd75b2304c9)

[Try this Model]{https://huggingface.co/spaces/Mr-Vicky-01/tamil_translator}
