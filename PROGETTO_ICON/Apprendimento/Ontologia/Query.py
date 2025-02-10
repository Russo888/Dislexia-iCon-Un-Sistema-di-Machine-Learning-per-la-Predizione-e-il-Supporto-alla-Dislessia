from owlready2 import *

# Carica l'ontologia
onto = get_ontology("ontologia_dislessia.owx").load()

# Stampa le classi, proprietà e individui
print("-----------------------Classi in ontologia:--------------------------\n")
print(list(onto.classes()), "\n")

print("-----------------------Proprietà oggetto:--------------------------\n")
print(list(onto.object_properties()), "\n")

print("-----------------------Proprietà dati:------------------------------\n")
print(list(onto.data_properties()), "\n")

# Cerca e stampa individui della classe Persona
print("-----------------------Lista delle persone:--------------------------\n")
persone = onto.search(is_a = onto.Persona)
print(persone, "\n")

# Cerca e stampa individui della classe Prestazioni
print("-----------------------Lista delle prestazioni:----------------------\n")
prestazioni = onto.search(is_a = onto.Prestazioni)
print(prestazioni, "\n")

# QUERY------------------------------------------------------------------------
print("_____________________________QUERY______________________________________\n")

# Trova le persone che hanno effettuato una prestazione
personeConPrestazioni = [p for p in onto.Persona.instances() if p.haPrestazione]

print("- Persone che hanno effettuato una prestazione:\n")
for persona in personeConPrestazioni:
    print(persona.name)


# Trova le persone che sono dislessiche
personeDislessiche = onto.search(is_a = onto.Persona, haDislessia=onto.Dislessia_Yes)

print("\n\n- Persone dislessiche:\n")
for persona in personeDislessiche:
    print(persona.name)
