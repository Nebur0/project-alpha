"""Motore di esecuzione: dai pesi target agli ordini, dietro il Risk Engine.

- ``orders``: pianificazione degli ordini e identificativi deterministici;
- ``broker``: interfaccia comune dei broker e broker simulato locale;
- ``engine``: ribilanciamento (rischio → piano → invio → riconciliazione).

Nessun adattatore verso broker reali è incluso: si aggiungono implementando
``Broker`` solo su richiesta esplicita del proprietario (vedi CLAUDE.md).
"""
