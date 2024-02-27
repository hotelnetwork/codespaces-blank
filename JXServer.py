import socket
import concurrent.futures

class Server:
    def __init__(self, host='localhost', port=8001):
        self.connections = []
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((host, port))
        self.server.listen()

    def handle_client(self, conn, addr):
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            for c in self.connections:
                if c != conn:
                    c.sendall(data)
        conn.close()
        self.connections.remove(conn)

    def start(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                conn, addr = self.server.accept()
                self.connections.append(conn)
                executor.submit(self.handle_client, conn, addr)

server = Server()
server.start()