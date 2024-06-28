# RU-GEE
Grammar Error Explanation and Russian Error Corpus


## Approach

To perform the task of Grammar Error Explanation, we employed an instruction tuning approach. We fine-tuned the `ruGPT3Small` model using labeled data.

## Data

We used synthetic data generated with the help of the PyMorphy to create erroneous sentences. The sentences were sourced from parts 1 and 2 of the "Привет, Россия" book. This data was standardized to a format suitable for instruction tuning with large language models (LLMs). Our dataset comprised sentences labeled as grammatically incorrect, along with explanations of the error types and their corrections.
