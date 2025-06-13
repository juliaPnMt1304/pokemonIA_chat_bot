from flask import Flask, request, jsonify, render_template
from agentes import criar_rag_chain_manual, avaliar_resposta, carregar_e_dividir_documentos, criar_index
import os

app = Flask(__name__, template_folder='.')
docs = carregar_e_dividir_documentos("arquivos")
index = criar_index(docs)
rag = criar_rag_chain_manual(index.as_retriever())

@app.route('/')
def home():
    return render_template("pokemon.html")

@app.route('/perguntar', methods=['POST'])
def perguntar():
    data = request.json
    pergunta = data.get("pergunta")

    if not pergunta:
        return jsonify({"erro": "Pergunta não fornecida"}), 400

    resposta = rag(pergunta)
    avaliacao = avaliar_resposta(pergunta, resposta['answer'])

    return jsonify({
        "resposta": resposta['answer'],
        "avaliacao": avaliacao
    })

if __name__ == '__main__':
    app.run(debug=True)
