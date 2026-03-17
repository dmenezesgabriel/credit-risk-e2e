c = get_config()

c.ServerApp.terminado_settings = {"shell_command": ["/bin/zsh"]}
c.ContentsManager.allow_hidden = True  # Show hidden files
c.IdentityProvider.token = ""
c.ServerApp.password = ""
