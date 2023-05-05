
# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
BERT_DIM = 768


VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN] + [f'speaker{x}' for x in range(1, 10)]

# dep labels
dep_i2s = [
    'CC',  # Coordinating conjunction
    'CD',  # Cardinal number
    'DT',  # Determiner
    'EX',  # Existential
    'FW',  # Foreign word
    'IN',  # Preposition or subordinating conjunction
    'JJ',  # Adjective
    'JJR',  # Adjective comparative
    'JJS',  # Adjective superlative
    'LS',  # List item marker
    'MD',  # Modal
    'NN',  # Noun singular
    'NNS',  # Noun plural
    'NNP',  # Proper noun singular
    'NNPS',  # Proper noun plural
    'PDT',  # Predeterminer
    'POS',  # Possessive ending
    'PRP',  # Personal pronoun
    'PRP$',  # Possessive pronoun
    'RB',  # Adverb
    'RBR',  # Adverb comparative
    'RBS',  # Adverb superlative
    'RP',  # Particle
    'SYM',  # symbol
    'TO',  # to
    'UH',  # Interjection
    'VB',  # verb base form
    'VBD',  # verb past tense
    'VBG',  # verb gerund or present participle
    'VBN',  # verb past participle
    'VBP',  # verb non-3rd person singular persent
    'VBZ',  # verb 3rd person singular persent
    'WDT',  # wh-determiner
    'WP',  # wh-pronoun
    'WP$',  # possessive wh-pronoun
    'WRB',  # wh-adverb
]

dep_s2i = {s:i for i,s in enumerate(dep_i2s)}

pos_i2s = [
    '', # blank token for padded sequence, added by Emrys
    'ADJ', #    adjective    big, old, green, incomprehensible, first
    'ADP', #   adposition    in, to, during
    'ADV', #     adverb    very, tomorrow, down, where, there
    'AUX', #     auxiliary    is, has (done), will (do), should (do)
    'CONJ', #     conjunction    and, or, but
    'CCONJ', #     coordinating conjunction    and, or, but
    'DET', #     determiner    a, an, the
    'INTJ', #     interjection    psst, ouch, bravo, hello
    'NOUN', #     noun    girl, cat, tree, air, beauty
    'NUM', #     numeral    1, 2017, one, seventy-seven, IV, MMXIV
    'PART', #     particle    ’s, not,
    'PRON', #     pronoun    I, you, he, she, myself, themselves, somebody
    'PROPN', #     proper noun    Mary, John, London, NATO, HBO
    'PUNCT', #     punctuation    ., (, ), ?
    'SCONJ', #     subordinating conjunction    if, while, that
    'SYM', #     symbol    $, %, §, ©, +, −, ×, ÷, =, :),
    'VERB', #     verb    run, runs, running, eat, ate, eating
    'X', #     other    sfpksdpsxmsa
    'SPACE', #     space
]

pos_s2i = {s:i for i,s in enumerate(pos_i2s)}


ner_i2s = [
    '', # blank token for none named entities
    'PERSON', #    People, including fictional.
    'NORP', #    Nationalities or religious or political groups.
    'FAC', #    Buildings, airports, highways, bridges, etc.
    'ORG', #    Companies, agencies, institutions, etc.
    'GPE', #    Countries, cities, states.
    'LOC', #    Non-GPE locations, mountain ranges, bodies of water.
    'PRODUCT', #    Objects, vehicles, foods, etc. (Not services.)
    'EVENT', #    Named hurricanes, battles, wars, sports events, etc.
    'WORK_OF_ART', #    Titles of books, songs, etc.
    'LAW', #    Named documents made into laws.
    'LANGUAGE', #    Any named language.
    'DATE', #    Absolute or relative dates or periods.
    'TIME', #    Times smaller than a day.
    'PERCENT', #    Percentage, including ”%“.
    'MONEY', #    Monetary values, including unit.
    'QUANTITY', #    Measurements, as of weight or distance.
    'ORDINAL', #    “first”, “second”, etc.
    'CARDINAL', #   Numerals that do not fall under another type.
]

ner_s2i = {s:i for i,s in enumerate(ner_i2s)}

label_i2s = [
    'per:positive_impression', # use underline to concatenate two words
    'per:negative_impression',
    'per:acquaintance',
    'per:alumni',
    'per:boss',
    'per:subordinate',
    'per:client',
    'per:dates',
    'per:friends',
    'per:girl/boyfriend',
    'per:neighbor',
    'per:roommate',
    'per:children',
    'per:other_family',
    'per:parents',
    'per:siblings',
    'per:spouse',
    'per:place_of_residence',
    'per:place_of_birth',
    'per:visited_place',
    'per:origin',
    'per:employee_or_member_of',
    'per:schools_attended',
    'per:works',
    'per:age',
    'per:date_of_birth',
    'per:major',
    'per:place_of_work',
    'per:title',
    'per:alternate_names',
    'per:pet',
    'gpe:residents_of_place',
    'gpe:births_in_place',
    'gpe:visitors_of_place',
    'org:employees_or_members',
    'org:students',
    'unanswerable',
]

label_s2i = {s:i for i,s in enumerate(label_i2s)}

