import select
import socket
from flask import Flask, request

app = Flask(__name__)

@app.route('/message', methods=['POST'])
def handle_message():
    user = request.form.get('user')
    msg = request.form.get('msg')
    # Here you can handle the message, for example, send it to other users
    print(f'Received message from {user}: {msg}')
    return 'Message received', 200

class Server:
    def __init__(self, host='localhost', port=8000):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen()
        self.server.setblocking(False)
        self.inputs = [self.server]
        self.outputs = []
        self.clients = {}

    def handle_client(self, conn):
        data = conn.recv(1024)
        if data:
            user, msg = data.decode().split(':', 1)
            if user in self.clients:
                self.clients[user].sendall(msg.encode())
            else:
                print(f'User {user} not found.')
        else:
            self.inputs.remove(conn)
            conn.close()

    def start(self):
        print("Server is starting...")
        while self.inputs:
            readable, writable, exceptional = select.select(self.inputs, self.outputs, self.inputs)
            for s in readable:
                if s is self.server:
                    conn, addr = s.accept()
                    conn.setblocking(False)
                    self.inputs.append(conn)
                    self.clients[addr[0]] = conn
                else:
                    self.handle_client(s)

server = Server()
server.start()

if __name__ == '__main__':
    app.run(host='localhost', port=8000)