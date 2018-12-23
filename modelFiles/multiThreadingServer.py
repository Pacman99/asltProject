import socket
from threading import Thread
import sys
import subprocess

class ClientThread(Thread):
   
   def __init__(self, ip, port):
      Thread.__init__(self)
      self.ip = ip
      self.port = port
      print("[+] New server socket thread started for " + ip+ ":" + str(port))

   def run(self):
      data = conn.recv(2048)
      print("Server received data:",data)
      cnnFileName = ""
      #cnnOutput = subprocess.check_output([sys.executable, cnnFileName, data])
      #Message = cnnOutput
      Message = "Hey"
      Message = Message.encode('utf-8')
      #cnnOutput might have to be cut and then passed to Message
      conn.send(Message)

TCP_IP = '0.0.0.0'
TCP_PORT = 2004
BUFFER_SIZE = 20
tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcpServer.bind((TCP_IP, TCP_PORT))
threads =[]
while True:
   tcpServer.listen(4)
   print("Multithreaded Python server : Waiting for connections from TCP clients..")
   (conn, (ip, port)) = tcpServer.accept()
   newthread = ClientThread(ip, port)
   newthread.run()
   threads.append(newthread)

for t in threads:
   t.join()
