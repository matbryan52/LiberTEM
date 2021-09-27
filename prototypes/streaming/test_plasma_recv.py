import socket
import numpy as np
import pyarrow as pa
import pyarrow.plasma as plasma
import time

HOST = '127.0.0.1'
PORT = 50007
MSGSIZE = 2**24
NMSG = 5
print(f'{MSGSIZE / 2**20} MB x {NMSG} messages')


def get_data(num=None):
    base_array = np.arange(255, dtype=np.uint8) if num is None else np.ones((1,), dtype=np.uint8) * num
    array = np.resize(base_array, MSGSIZE)
    return array.tobytes()


def get_buf(client):
    object_id = plasma.ObjectID.from_random()
    buf = client.create(object_id, MSGSIZE)
    return object_id, buf


# @profile
# def recv(client, conn):
#     print('Recieving')
#     object_id, buf = get_buf(client)
#     nbytes = conn.recv_into(buf, MSGSIZE)
#     return nbytes
#     if not nbytes:
#         return False
#     client.seal(object_id)

#     buffers = client.get_buffers([object_id])
#     array = np.frombuffer(buffers[0], dtype=np.uint8)
#     print(array[0])
#     client.delete([object_id])
#     del object_id
#     return True

@profile
def recv(conn, client):
    msg_count = 0
    while True:
        if msg_count == 0:
            object_id, buf = get_buf(client)
        nbytes = conn.recv_into(buf, MSGSIZE)
        if not nbytes:
            break
        msg_count += nbytes
        if msg_count < MSGSIZE:
            continue
        if not nbytes:
            break

        print(f'Recieved {msg_count // 2**20} MB')
        msg_count = 0
        client.seal(object_id)
        buffers = client.get_buffers([object_id])
        array = np.frombuffer(buffers[0], dtype=np.uint8)
        print(f'Message number: {np.unique(array)}')
        client.delete([object_id])
        del object_id


if __name__ == '__main__':
    client = plasma.connect("/tmp/plasma")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        conn, addr = s.accept()
        print(f'Connected {addr}')
        with conn:
            recv(conn, client)

