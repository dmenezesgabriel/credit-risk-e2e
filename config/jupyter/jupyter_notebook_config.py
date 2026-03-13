c = get_config()

# Use ServerApp for modern JupyterLab/Jupyter Server versions
c.ServerApp.terminado_settings = {"shell_command": ["/bin/zsh"]}
