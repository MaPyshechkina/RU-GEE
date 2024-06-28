# -*- coding: utf-8 -*-
"""Gen_data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1b1OATHZcA4rKQj9f3bpPoHZCau-zG3GI
"""



import pymorphy2
import random
import re
import pandas as pd


morph = pymorphy2.MorphAnalyzer()


cases = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']

def inflect_random_case(word, current_case):

    parsed_word = morph.parse(word)[0]
    possible_cases = [case for case in cases if case != current_case]
    new_case = random.choice(possible_cases)
    inflected_word = parsed_word.inflect({new_case})
    return inflected_word.word if inflected_word else word

def get_case(word):

    parsed_word = morph.parse(word)[0]
    return parsed_word.tag.case

def introduce_error(sentence):

    words = sentence.split()
    random.shuffle(words)

    for word in words:
        word_clean = re.sub(r'\W+', '', word)
        current_case = get_case(word_clean)

        if current_case and current_case in cases:
            new_word = inflect_random_case(word_clean, current_case)
            if new_word != word_clean:
                erroneous_sentence = sentence.replace(word_clean, new_word, 1)
                return sentence, erroneous_sentence, word_clean, new_word

    return sentence, sentence, None, None

def process_text(input_text):

    sentences = input_text.split('\n')
    result = []

    for sentence in sentences:
        original_sentence, erroneous_sentence, original_word, changed_word = introduce_error(sentence)
        result.append({
            "Original Sentence": original_sentence,
            "Erroneous Sentence": erroneous_sentence,
            "Original Word": original_word,
            "Changed Word": changed_word
        })

    return result


input_text = """Ахмет: А я много работаю и уже давно не был в отпуске.
Мой самый интересный отпуск был в Сочи, когда там были Олимпийские игры.
Это русский город на юге.
Турция близко.
Море тёплое.
Я был в парке «Ривьера», на горе Ахун, видел очень красивые соборы, был в океанариуме.
А ещё здесь есть очень красивое место — Красная Поляна.
Осенью там все деревья красные.
Как было хорошо!
Сакура: Я отдыхала летом, потому что наши студенты тоже отдыхали.
Я была в Хабаровске.
Этот русский город находится на востоке, не очень далеко от Токио.
Я видела «Амурское чудо» — это большой мост на реке Амур.
Потом была в соборе, гуляла на площади.
А ещё я видела Амурские столбы — это так красиво!
Природа долго создавала эти огромные каменные столбы.
Они стоят здесь очень давно.
Клаус: Вы знаете, что я учусь в университете и работаю в кафе и не могу долго отдыхать.
Но недавно я был в Риме.
Там живёт мой друг."""

output = process_text(input_text)

df = pd.DataFrame(output)


df.to_excel('processed_sentences.xlsx', index=False)

from google.colab import files
files.download('processed_sentences.xlsx')

