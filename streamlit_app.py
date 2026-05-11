import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os

st.set_page_config( 
    page_title="Translitor"
)

st.image("english_german_flag.png")

st.markdown(
    '<h1 style="text-align:center">Translitor</h1>',
    unsafe_allow_html=True
)

st.write("")

with st.expander(":bulb: Informacje", expanded=False):
    st.markdown("""
        Translitor pozwala na użycie dwóch narzędzi opartych na modelach Huggin Face:
        
        * :red[Wydźwięk emocjonalny tekstu] - analizuje czy tekst jest pozytywny dla języka angielskiego.
        * :yellow[Tłumacz angielskiego na niemiecki] - tłumaczy tekst z języka angielskiego na niemiecki. 
        
        **Instrukcja**
        1. Wybierz metodę z podanych dwóch opcji.
        2. Wpisz tekst w polu.
        3. Poczekaj na wynik
        4. Ciesz się odpowiedzią :sunflower:
    """
    )

st.write("")

st.header('Przetwarzanie języka naturalnego')


import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumacz angielskiego na niemiecki",
    ],
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst (wydźwięk emocjonalny)")
    if text:
        try:
            with st.spinner("Analiza wydźwięku..."):
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)
            
            label = answer[0]["label"]
            score = answer[0]["score"]

            st.write("")
            st.write("")
            st.subheader("Analiza wydźwięku emocjonalnego")
            if label == "POSITIVE":
                st.success(f"Wynik: {label} Dokładność : {score}")
            else:
                st.error(f"Wynik: {label} Dokładność: {score}")

            with st.expander("Szczegóły odpowiedzi"):
                st.write(answer)

        except Exception as e:
            st.error(f"Błąd podczas analizy: {e}")

if option == "Tłumacz angielskiego na niemiecki":
    text = st.text_area(label="Wpisz tekst (tłumacz angielskiego na niemiecki)")
    if text:
        try:
            with st.spinner("Tłumaczenie tekstu na język niemiecki..."):
                tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
                model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
                
                inputs = tokenizer(text, return_tensors="pt")
                outputs = model.generate(**inputs)
                translation = tokenizer.decode(outputs[0], skip_special_tokens=True)


            st.write("")
            st.write("")

            st.subheader("Tłumaczenie tekstu z języka angielskiego na niemiecki")
            st.success("Tłumaczenie zakończone!")


            col1, col2 = st.columns(2)

            with col1:
                st.markdown(":red[Angielski]")
                st.text(text)
            with col2:
                st.markdown(":yellow[Niemiecki]")
                st.text(translation)
            

        except Exception as e:
            st.error(f"Błąd podczas tłumaczenia: {e}")




st.divider()
st.markdown('<p style="text-align:center; color:gray;">' 
    'Nr indeksu: s29504'
    '</p>', 
    unsafe_allow_html=True
)


