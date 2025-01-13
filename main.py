from flask import Flask, render_template, request, Response, jsonify
import base64
import ollama
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image_data = request.json['image']
        
        def generate():
            try:
                response = ollama.chat(
                    model="llava:7b",
                    messages=[
                        {
                            'role': 'user',
                            'content': 'Describe this image in detail:',
                            'images': [image_data]
                        }
                    ],
                    stream=True
                )
                for chunk in response:
                    if chunk['message']['content']:
                        yield chunk['message']['content']
            except Exception as e:
                yield f"Error during analysis: {str(e)}"
        
        return Response(generate(), mimetype='text/plain')
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
