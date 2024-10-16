# RU-GEE
Grammar Error Explanation and Russian Error Corpus


## Approach

To perform the task of Grammar Error Explanation, we employed an instruction tuning approach. We fine-tuned the `ruGPT3Small` model using labeled data.

## Data

We used synthetic data generated with the help of the PyMorphy to create erroneous sentences. The sentences were sourced from parts 1 and 2 of the "Привет, Россия" book. This data was standardized to a format suitable for instruction tuning with large language models (LLMs). Our dataset comprised sentences labeled as grammatically incorrect, along with explanations of the error types and their corrections.

## Synthtic Corpora

| Case | n   |
|------|-----|
| Nom  | 574 |
| Acc  | 232 |
| Loc  | 207 |
| Gen  | 101 |
| Ins  | 55  |
| Dat  | 37  |
| Total| 1206|

|     | Noun | Adj | Pron | Num | Adv |
|-----|------|-----|------|-----|-----|
| Nom | 348  | 144 | 72   | 10  |     |
| Acc | 144  | 40  | 42   | 6   |     |
| Loc | 164  | 24  | 14   | 5   | 2   |
| Gen | 80   | 14  | 4    | 1   |     |
| Ins | 33   | 17  | 2    | 3   |     |
| Dat | 26   | 3   | 7    | 1   |     |
| Total | 795 | 242 | 141  | 26  |     |
