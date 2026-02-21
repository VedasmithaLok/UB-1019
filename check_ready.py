import os
import sys

print("=== Hackathon Project Checklist ===\n")

checks = []

# Check 1: Model file
if os.path.exists('xray_model.pth'):
    print("✓ Model file found")
    checks.append(True)
else:
    print("✗ Model file missing - Run train_fast.py first")
    checks.append(False)

# Check 2: Backend files
backend_files = ['backend/model.py', 'backend/report_generator.py', 
                 'backend/translator.py', 'backend/voice.py',
                 'backend/utils/qr_generator.py', 'backend/gradcam.py']

all_backend = True
for f in backend_files:
    if not os.path.exists(f):
        print(f"✗ Missing: {f}")
        all_backend = False

if all_backend:
    print("✓ All backend files present")
checks.append(all_backend)

# Check 3: Main app
if os.path.exists('app.py'):
    print("✓ Main app file found")
    checks.append(True)
else:
    print("✗ app.py missing")
    checks.append(False)

# Check 4: Requirements
if os.path.exists('requirements.txt'):
    print("✓ Requirements file found")
    checks.append(True)
else:
    print("✗ requirements.txt missing")
    checks.append(False)

# Check 5: __init__ files
init_files = ['backend/__init__.py', 'backend/utils/__init__.py']
all_init = True
for f in init_files:
    if not os.path.exists(f):
        all_init = False

if all_init:
    print("✓ Package init files present")
else:
    print("⚠ Some __init__.py files missing (may cause import errors)")
checks.append(all_init)

print("\n" + "="*40)
if all(checks):
    print("✓ ALL CHECKS PASSED - Ready for hackathon!")
    print("\nTo run: streamlit run app.py")
else:
    print("✗ Some checks failed - Fix issues above")
    sys.exit(1)
