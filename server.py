from flask import Flask, jsonify
from flask_pydantic import validate
from pydantic import BaseModel, Field
from rag import rag_query
from misc import make_joke, make_invitation

app = Flask(__name__)

class QuestionQuery(BaseModel):
    question: str = Field(..., min_length=1)

@app.route('/answer', methods=['GET'])
@validate(query=QuestionQuery)
def answer_route(query: QuestionQuery):
    answer, contexts = rag_query(query.question)
    
    contexts_data = []
    if contexts:
        for doc in contexts:
            contexts_data.append({
                "metadata": getattr(doc, 'metadata', {}),
                "page_content": getattr(doc, 'page_content', str(doc))
            })
            
    return jsonify({
        "answer": answer,
        "contexts": contexts_data
    })

@app.route('/joke', methods=['GET'])
@validate(query=QuestionQuery)
def joke_route(query: QuestionQuery):
    joke_text = make_joke(query.question)
    return jsonify({
        "joke": joke_text
    })

@app.route('/invitation', methods=['GET'])
@validate(query=QuestionQuery)
def invitation_route(query: QuestionQuery):
    answer, _ = rag_query(query.question)
    invitation_text = make_invitation(answer)
    return jsonify({
        "invitation": invitation_text
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
