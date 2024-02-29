import socket

class StaticClient:
    @staticmethod
    def connect(host='localhost', port=8000):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        return sock

    @staticmethod
    def send_receive(sock):
        while True:
            user_input = input("Enter a message to send to the server: ")
            if user_input.lower() == 'quit':
                break
            sock.sendall(user_input.encode())
            data = sock.recv(1024)
            print('Received from server:', repr(data.decode()))

    @staticmethod
    def close(sock):
        sock.close()

# Connect to the server
sock = StaticClient.connect(port=int(input("Enter the port number to connect to: ")))

# Send and receive data
StaticClient.send_receive(sock)

# Close the connection
StaticClient.close(sock)