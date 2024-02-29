import socket

class JXClient:
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.sock.connect((self.host, self.port))

    def send_receive(self):
        while True:
            user_input = input("Enter a message to send to the server: ")
            if user_input.lower() == 'quit':
                break
            self.sock.sendall(user_input.encode())
            data = self.sock.recv(1024)
            print('Received from server:', repr(data.decode()))

    def close(self):
        self.sock.close()

# Create a client object
client = JXClient(port=int(input("Enter the port number to connect to: ")))

# Connect to the server
client.connect()

# Send and receive data
client.send_receive()

# Close the connection
client.close()