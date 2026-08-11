from datetime import datetime, timedelta
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_logs import main

def esperar_hasta_las_23():
    """Espera en bucle hasta que el reloj marque las 23:00."""
    while datetime.now().hour != 13:
        print("en espera...")
        time.sleep(30)

esperar_hasta_las_23()
print("inciando...")
for segundo in range(10, 0, -1):
    print(segundo)
    if segundo == 3:
        print("quedan 3 segundos...")
        os.system("paplay window-attention.oga") # <----- pitido avisando que faltan 3 segundos para iniciar main
    time.sleep(1)
main()