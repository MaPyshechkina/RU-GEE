# как автоматически определить падеж
import pandas as pd
import spacy


nlp = spacy.load('ru_core_news_sm')

def get_case(token):
    if token.pos_ in ['NOUN', 'ADJ', 'PRON', 'NUM', 'VERB', 'PART', 'DET', 'PROPN']:
        case = token.morph.get('Case')
        if case:
            return case[0]
    return None

def determine_case(row):
    correct_word = row['Исправление']
    corrected_sentence = row['Исправленное предложение']


    if pd.isna(correct_word) or pd.isna(corrected_sentence):
        return None

    correct_word = str(correct_word).strip()
    corrected_sentence = str(corrected_sentence).strip()


    correct_word_clean = correct_word.replace("*", "").strip()
    corrected_sentence_clean = corrected_sentence.replace("*", "").strip()

    doc = nlp(corrected_sentence_clean)
    matched_token = None
    for token in doc:

        if token.text.lower() == correct_word_clean.lower():
            matched_token = token
            break

        if correct_word_clean.lower() in token.text.lower():
            matched_token = token
            break

    if matched_token:
        case = get_case(matched_token)
        if case:
            return case
        else:
            return "Case not found"
    return "No match found"


input_file_path = ' PrivetRo_mistakes.xlsx'


df = pd.read_excel(input_file_path)


df['Исправление'] = df['Исправление'].astype(str).fillna('')
df['Исправленное предложение'] = df['Исправленное предложение'].astype(str).fillna('')


df['Тип ошибки'] = df.apply(determine_case, axis=1)


output_file_path = 'PR_mistakes.xlsx'


df.to_excel(output_file_path, index=False)

print(f"Обработанные данные сохранены в '{output_file_path}'")

