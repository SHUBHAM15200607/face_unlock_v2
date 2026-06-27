import subprocess

SESSION_ID = "2"

def unlock():

    print("Unlocking Session...")

    subprocess.run(
        ["loginctl", "unlock-session", SESSION_ID],
        check=False
    )
