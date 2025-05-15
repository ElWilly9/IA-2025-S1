import termcolor
from logic import *

Plum = Symbol("Plum")
Green = Symbol("Green")
Mustard = Symbol("Mustard")
Peacock = Symbol("Peacock")
characters = [Plum, Green, Mustard, Peacock]

estudio = Symbol("estudio")
pasillo = Symbol("pasillo")
sala = Symbol("sala")
juegos = Symbol("juegos")
rooms = [estudio, pasillo, sala, juegos]

revolver = Symbol("revolver")
hacha = Symbol("hacha")
candelabro = Symbol("candelabro")
herramienta = Symbol("herramienta")
weapons = [revolver, hacha, candelabro, herramienta]

symbols = characters + rooms + weapons


def check_knowledge(knowledge):
    for symbol in symbols:
        if model_check(knowledge, symbol):
            termcolor.cprint(f"{symbol}: SI", "green")
        elif not model_check(knowledge, Not(symbol)):
            print(f"{symbol}: TAL VEZ")


# There must be a person, room, and weapon.
knowledge = And(
    Or(Plum, Green, Mustard, Peacock),
    Or(estudio, pasillo, sala, juegos),
    Or(revolver, hacha, candelabro, herramienta)
)


print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)

# Initial cards
knowledge.add(And(
    Not(Plum), Not(estudio), Not(revolver)
))
print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)

# Unknown card
knowledge.add(Or(
    Not(Peacock), Not(estudio), Not(revolver)
))
print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)

# Known cards
knowledge.add(Not(Peacock))
knowledge.add(Not(herramienta))
print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)

# Unknown card
knowledge.add(Or(
    Not(juegos), Not(estudio), Not(Mustard)
))
print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)

# Unknown card
knowledge.add(Or(
    Not(Peacock), Not(juegos), Not(revolver)
))
print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)

# Unknown card
knowledge.add(Not(Green))
print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)

# Unknown card
knowledge.add(Not(candelabro))
print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)

# Unknown card
knowledge.add(And(Not(sala), Not(estudio)))
print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)

# Unknown card
knowledge.add(Not(juegos))
print("Base de conocimiento: ", knowledge.formula())
print("¿Qué sé?")
check_knowledge(knowledge)
