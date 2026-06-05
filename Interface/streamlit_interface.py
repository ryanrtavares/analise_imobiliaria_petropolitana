import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Carregamento dos arquivos (mantive seu caminho)
with open(r"C:\Users\Ryan\OneDrive\Documentos\GitHub\analise_imobiliaria_petropolitana\pickle_xgboost", "rb") as modelo_file:
    modelo = pickle.load(modelo_file)

with open(r"C:\Users\Ryan\OneDrive\Documentos\GitHub\analise_imobiliaria_petropolitana\pickle_label", "rb") as label_file:
    le = pickle.load(label_file)

st.title("Ultra House Price Prediction")

st.header("Insira os dados do imóvel:")
tamanho = st.number_input("Tamanho (m²) do imóvel: ", min_value=0.0)
quartos = st.number_input("Quantidade de quartos: ", min_value=0, step=1)
vagas = st.number_input("Quantidade de vagas: ", min_value=0, step=1)
banheiros = st.number_input("Quantidade de banheiros: ", min_value=0, step=1)
iptu = st.number_input("Valor do IPTU: ", min_value=0.0)
condominio = st.number_input("Condomínio: ", min_value=0.0)
bairro = st.selectbox("Selecione seu bairro",[
    "Itaipava", "Centro", "Pedro do Rio", "Corrêas", "Araras", 
    "Quitandinha", "Nogueira", "Mosela", "Retiro", "Samambaia", 
    "Valparaíso", "Castelanea", "Bingen", "Independência", "Coronel Veiga", 
    "Cuiabá", "Alto da Serra", "São Sebastião", "Bonsucesso", "Fazenda Inglesa", 
    "Morin", "Saldanha Marinho", "Carangola", "Duarte Silveira", "Quarteirão Ingelheim", 
    "Posse", "Cascatinha", "Itamarati", "Quarteirão Brasileiro", "Quissama", 
    "Duchas", "Simeria", "Caxambu", "Chácara Flora", "Estrada da Saudade", 
    "Castrioto", "Floresta", "Gloria", "Vale do Carangola", "Duques"
]) # (Sua lista completa aqui)

if st.button("Prever Preço"):
    bairro_codificado = le.transform([bairro])[0]

    # 1. Crie o dicionário com os dados
    dados_input = {
        'm2': [tamanho],
        'quartos': [quartos],
        'vagas': [vagas],
        'banheiros': [banheiros],
        'iptu': [iptu],
        'condominio': [condominio],
        'bairro': [bairro_codificado]
    }

    # 2. Crie o DataFrame
    input_df = pd.DataFrame(dados_input)

    # 3. GARANTIA: Force o DataFrame a ter a ordem exata das colunas que o modelo espera
    colunas_esperadas = ['m2', 'quartos', 'vagas', 'banheiros', 'iptu', 'condominio', 'bairro']
    input_df = input_df[colunas_esperadas]

    # 4. Agora pode prever sem erro de mismatch
    pred = modelo.predict(input_df)
    
    st.success(f"Preço estimado: R$ {pred[0]:,.2f}")