import socket
import concurrent.futures

def handle_client(conn, addr):
    print(f"Connected by {addr}")
    while True:
        # data = conn.recv(1024)
        # Accept user input and send it back to the client
        data = input("Enter a message: ").encode()
        # If the user enters an empty string, break the loop
        if not data:
            break
        data = "Jinx Systems: " + data.decode()
        conn.sendall(data.encode())
    conn.close()

def start_server(host='localhost', port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                conn, addr = s.accept()
                executor.submit(handle_client, conn, addr)

start_server()