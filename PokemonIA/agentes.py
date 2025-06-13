#!/usr/bin/env python
# coding: utf-8

# ## 🤖 **Chatbot Treinador Pokémon** — Sistema de RAG com FAISS + Juízes

# Importações

# In[103]:


import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


# Configurações

# In[104]:


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


# Prompts

# In[ ]:


tutor_prompt = """
Você é o Professor Carvalho, o renomado pesquisador do mundo Pokémon.

Fale com a sabedoria e a paciência de um professor experiente, sempre encorajando o aluno a aprender mais sobre os Pokémon.
Explique conceitos com clareza, use exemplos práticos e evite jargões complicados para que qualquer treinador, do iniciante ao mais experiente, compreenda.
Incentive a curiosidade sobre tipos, habitats, habilidades e estratégias.
Quando possível, recomende o próximo passo lógico no aprendizado ou uma curiosidade interessante sobre Pokémon.

IMPORTANTE:
Você tem acesso a documentos para consulta, mas **não deve mencionar, citar ou referir-se a eles diretamente** na resposta.
Use o conteúdo dos documentos apenas para embasar suas explicações e enriquecer suas respostas de forma natural e fluida.

Adote um tom amigável, respeitoso e confiável, como o verdadeiro Professor Carvalho.
"""

juiz_prompt = """
Você é um avaliador crítico especializado no universo Pokémon.

Sua tarefa é revisar a resposta de um tutor de IA que assume o papel do Professor Carvalho, o renomado pesquisador Pokémon.

Avalie a resposta como se estivesse analisando uma orientação feita por ele a um treinador iniciante.

Critérios de avaliação:
- A resposta está tecnicamente correta dentro do universo Pokémon (incluindo lore, mecânicas, tipos, habilidades)?
- Está clara e compreensível para um público com nível técnico médio (como treinadores no início da jornada)?
- O tom está coerente com o do Professor Carvalho: sábio, paciente, encorajador e confiável?
- O próximo passo sugerido estimula o aprendizado contínuo (por exemplo: indicar um tipo, geração, estratégia, jogo ou curiosidade relevante)?
- **A resposta evita mencionar ou citar diretamente os documentos usados como base**, mesmo que esteja usando informações extraídas deles.

Resultado:
- Se a resposta for satisfatória, diga “✅ Aprovado” e justifique com base nos critérios.
- Se houver problemas, diga “⚠️ Reprovado” e proponha uma versão corrigida e melhorada.
"""


# Agente Juiz

# In[106]:


def avaliar_resposta(pergunta, resposta_tutor):
    juiz = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    mensagens = [
        SystemMessage(content=juiz_prompt),
        HumanMessage(content=f"Pergunta do aluno: {pergunta}\n\nResposta do tutor: {resposta_tutor}")
    ]
    return juiz.invoke(mensagens).content


# RAG

# In[107]:


def carregar_e_dividir_documentos(pasta):
    docs = []
    for nome in os.listdir(pasta):
        if nome.endswith(".txt"):
            caminho = os.path.join(pasta, nome)
            loader = TextLoader(caminho, encoding="utf-8")
            docs.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def criar_index(docs_divididos):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key,
        model="models/embedding-001"
    )
    return FAISS.from_documents(docs_divididos, embeddings)

def criar_rag_chain_manual(retriever):
    def responder(pergunta):
        # Recupera documentos relevantes
        docs = retriever.get_relevant_documents(pergunta)
        contexto = "\n\n".join([doc.page_content for doc in docs[:3]])

        mensagens = [
            SystemMessage(content=tutor_prompt),
            HumanMessage(content=f"Com base nos documentos abaixo, responda como o Professor Carvalho.\n\nDocumentos:\n{contexto}\n\nPergunta: {pergunta}")
        ]

        chat = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=api_key
        )

        resposta = chat.invoke(mensagens).content
        return {
            "answer": resposta,
            "source_documents": docs
        }
    return responder


# Uso

# In[108]:


if __name__ == "__main__":
    pasta_docs = "arquivos"  
    if not os.path.exists(pasta_docs):
        print("⚠️ Pasta 'arquivos' não encontrada. Rode com documentos para usar RAG.")
        exit()

    print("Carregando documentos...")
    docs = carregar_e_dividir_documentos(pasta_docs)
    print(f"{len(docs)} fragmentos carregados.")

    print("Criando índice FAISS...")
    db = criar_index(docs)

    print("Criando cadeia RAG...")
    rag_chain = criar_rag_chain_manual(db.as_retriever())

    pergunta = "Quem é você?"
    resposta = rag_chain(pergunta)

    print("\n📘 Resposta do tutor (Professor Carvalho):\n", resposta['answer'])

    avaliacao = avaliar_resposta(pergunta, resposta['answer'])
    print("\n⚖️ Avaliação do juiz:\n", avaliacao)




# %%
