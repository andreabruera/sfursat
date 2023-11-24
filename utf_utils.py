def transform_german_word(word):
    word = word.replace('ae', 'ä')
    word = word.replace('oe', 'ö')
    word = word.replace('ue', 'ü')
    word = word.replace('ss', 'ß')
    word = word.lower()
    return word
