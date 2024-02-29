from flask import Flask
import requests

app = Flask(__name__)

@app.route('/send_message/<user>/<msg>')
def send_message(user, msg):
    url = 'http://localhost:8000/message'
    data = {'user': user, 'msg': msg}
    response = requests.post(url, data=data)
    return 'Response from server: ' + response.text

if __name__ == '__main__':
    app.run(host='localhost', port=8001)