EIGENFLUID / METABALL - LOCAL INFERENCE

Windows quick start
1. Extract the ZIP to a normal writable folder.
2. Double-click START_LOCAL.bat.
3. The launcher uses py -3 when available, otherwise it automatically uses python.
4. The first launch creates a private Python environment and installs NumPy.
5. The browser opens http://127.0.0.1:8780 automatically.

The inference server binds only to 127.0.0.1. It does not expose the service
to the LAN, Tailscale, or the public Internet. After the first dependency
installation, inference and visualization run locally from the packaged FP32
checkpoints. Keep the command window open while using the application.
