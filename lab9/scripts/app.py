from flask import Flask, request, jsonify, render_template
import os
from driver import build_knowledgebase

app = Flask(__name__)

UPLOAD_FOLDER = '../data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

qa_chain = None  # 用于存储QA链，避免每次都重建

# 首页路由，渲染 HTML 页面
@app.route('/')
def index():
    return render_template('index.html')

# PDF 上传接口
@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    global qa_chain
    if 'pdfs' not in request.files:
        return jsonify({"message": "No files uploaded"}), 400

    files = request.files.getlist('pdfs')
    for file in files:
        if file.filename.endswith('.pdf'):
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))

    qa_chain = build_knowledgebase()
    return jsonify({"message": "PDFs uploaded and analyzed successfully."})

# 问答接口
@app.route('/chat', methods=['POST'])
def chat():
    if qa_chain is None:
        return jsonify({"reply": "Please upload PDF documents first."})

    data = request.json
    user_input = data.get('message', '')

    answer = qa_chain.invoke({"query": user_input})

    if isinstance(answer, dict):
        answer = answer.get('result', 'No answer found.')  # 取出文本部分
    elif not isinstance(answer, str):
        answer = str(answer)
        
    return jsonify({"reply": answer})

if __name__ == "__main__":
    app.run(debug=True)