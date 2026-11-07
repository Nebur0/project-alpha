"""Motore di esecuzione: dai pesi target agli ordini, dietro il Risk Engine.

- ``orders``: pianificazione degli ordini e identificativi deterministici;
- ``broker``: interfaccia comune dei broker e broker simulato locale;
- ``engine``: ribilanciamento (rischio → piano → invio → riconciliazione);
- ``alpaca_paper``: adattatore Alpaca **solo paper** (richiesto dal
  proprietario il 2026-11-07). Nessun adattatore verso conti reali: va
  aggiunto solo su richiesta esplicita del proprietario (vedi CLAUDE.md).
"""
