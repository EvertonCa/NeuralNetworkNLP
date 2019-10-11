# https://medium.com/parrot-prediction/dive-into-wordnet-with-nltk-b313c480e788
# http://clubedosgeeks.com.br/text-minning/wordnet-com-nltk

from nltk.corpus import wordnet as wn

termo = input()

syn = wn.synsets(termo)

for synonyms in syn:
    for synonym in synonyms.lemmas():
        print(synonym.name())


print("Done")