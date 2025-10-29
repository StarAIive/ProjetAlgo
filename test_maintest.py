import subprocess
import sys

# Simuler les entrées : mode 0, puis fichier 1 (A-n32-k5.vrp)
inputs = "0\n1\n"

try:
    # Exécuter le programme principal avec les entrées simulées
    process = subprocess.Popen([sys.executable, "maintest.py"], 
                             stdin=subprocess.PIPE, 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, 
                             text=True)
    
    stdout, stderr = process.communicate(input=inputs, timeout=120)
    
    print("=== SORTIE DU PROGRAMME ===")
    print(stdout)
    if stderr:
        print("=== ERREURS ===")
        print(stderr)
        
except subprocess.TimeoutExpired:
    process.kill()
    print("Le programme a dépassé le temps limite (120s)")
except Exception as e:
    print(f"Erreur lors de l'exécution: {e}")