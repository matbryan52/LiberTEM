import socket
from test_plasma_recv import HOST, PORT, NMSG, get_data


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        for num in range(NMSG):
            data = get_data(num=num)
            print(f'Sending {num}')
            s.send(data)


if __name__ == '__main__':
    main()
