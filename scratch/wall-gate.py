import os, subprocess, sys, time
print("GATE: waiting for a quiet window (load1 < 3.0 twice in a row) before the wall battery", flush=True)
quiet_streak = 0
while True:
    l = os.getloadavg()[0]
    quiet_streak = quiet_streak + 1 if l < 3.0 else 0
    print("gate tick load1=%.1f streak=%d %s" % (l, quiet_streak, time.strftime('%H:%M:%S')), flush=True)
    if quiet_streak >= 2:
        break
    time.sleep(60)
print("quiet window detected — launching wall battery %s" % time.strftime('%H:%M:%S'), flush=True)
rc = subprocess.run([sys.executable, os.path.join(os.getcwd(), "wall.py")], cwd=os.getcwd()).returncode
print("wall rc", rc, flush=True)
