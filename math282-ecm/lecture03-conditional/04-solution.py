def add_exclamation(sentence):
    already_there = sentence[-1]=="!"
    if not already_there:
        sentence = sentence + "!"
    return sentence


def add_exclamation(sentence):
    if sentence[-1] != "!":
        sentence = sentence + "!"
    return sentence

