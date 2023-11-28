import re

def transform_german_word(word, model):
    word = word.lower()
    word = re.sub('^ein\s|^eine\s|^der\s|^das\s|^die\s|^ne\s|^dann\s', '', word)
    word = re.sub('^e\s', 'e-', word)
    versions = [word]
    original_versions = [word]
    for word in original_versions:
        ### collecting different versions of a word
        if 'ae' in word:
            ae_versions = [w for w in versions]
            for w in ae_versions:
                corr_word = w.replace('ae', 'ä')
                versions.append(corr_word)
        if 'oe' in word:
            oe_versions = [w for w in versions]
            for w in oe_versions:
                corr_word = w.replace('oe', 'ö')
                versions.append(corr_word)
        if 'ue' in word:
            ue_versions = [w for w in versions]
            for w in ue_versions:
                corr_word = w.replace('ue', 'ü')
                versions.append(corr_word)
        if 'ss' in word:
            ss_versions = [w for w in versions]
            for w in ss_versions:
                corr_word = w.replace('ss', 'ß')
                versions.append(corr_word)
    versions = set(
                   [w for w in versions if w in model.keys()] + \
                   [w.capitalize() for w in versions if w in model.keys()] +\
                   [word.capitalize()] + \
                   [word],
                   )
    return versions
