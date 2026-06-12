# Lesson 13: Hermes Agent

This is my very opinionated hermes setup:

- `curator.enabled=false`: I'm not a fan of hermes editing my skills. I run the show here Herm.
- `/skills` are mounted readonly
- `config.yaml` is mounted readonly. This does prevent you from editing the config in the hermes Dashboard. 
- `hindsight` for memory with `"auto_retain": false, "auto_recall": false` set to disable memory being automaticaly saved and recalled for every prompt
- `camofox` for web scraping

## Start
```
cd hermes
cp .env.example .env
docker compose up -d
```

Add to your ~/.bashrc for a `hermes` command alias on your docker host
```
hermes() {
  docker exec -it hermes /opt/hermes/.venv/bin/hermes "$@"
}
```

Run the `hermes` command to launch the TUI or browse to the hermes Dashboard at http://gb10-ip:9119

## Adding Skills

The `config.yaml` is configured to read skills from the /skills directory. 

1. `cd` to the directory above `hermes` and run `git clone <github_repo>` 
2. Add a docker volume mapping to your hermes `compose.yaml`
```
hermes:
    volumes:
      - ../<skills_directory>:/skills/<skills_directory>:ro
```

## Adding MCP Servers

The `mcp_servers` section of `config.yaml` is where you add MCP servers. You can add as many as you want. 

https://hermes-agent.nousresearch.com/docs/reference/mcp-config-reference

### Via File (stdio)

Think of `stdio` like the `|` pipe command.

1. `cd` to the directory above `hermes` and run `git clone <github_repo>` 
2. Add a docker volume mapping to your hermes `compose.yaml`
```
hermes:
    volumes:
      - ../<mcp_server_directory>:/mcp/<mcp_server_directory>:ro
```
3. Add the `mcp_servers:` config for each MCP server. Some `stdio` MCP servers require environment variables to be declared. 
```
mcp_servers:
  jellyfin:
    command: "/opt/venv/bin/python"
    args: ["/mcp_servers/jellyfin.py"]
    env:
      JELLYFIN_URL: ${JELLYFIN_URL}
      JELLYFIN_API_KEY: ${JELLYFIN_API_KEY}
      JELLYFIN_USERNAME: ${JELLYFIN_USERNAME}
```
4. Since the hermes container is running these directly you will need to add their dependencies into the `requirements.txt` file and run `docker compose build hermes` to rebuild the container. 

### Via URL

```
mcp_servers:
  home_assistant:
    url: http://192.168.0.137:8124/mcp
    tools:
      include: [ha_get_state, ha_get_history]
```

The `include:` value is a list of the tools to include. Used to filter out tools from server.

## MCP Server Troubleshooting
```
docker exec -it hermes /bin/bash

cd /mcp_servers
fastmcp list arr/arr.py
fastmcp call arr/arr.py lookup_series term="Your Friends and Neighbors"
```

## Hindsight

https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/hindsight/README.md#tools

Running `hermes memory setup` and going through the hindsight install will fix the hindsight_client import failing. It doesn't need to save the config for it to install the dep.
I think the subagent has a separate pip PATH. Installing hindsight-client in the Dockerfile doesn't fix it. 

Add the docker volume mapping for the Hindsight config file: 

```
volumes:
  - ./hindsight.config.json:/opt/data/hindsight/config.json:ro
```

Update hermes `config.yaml`:
```
memory:
   provider: hindsight
```

Start the hindsight container:
```
docker compose --profile hindsight up -d
```

Hindsight Dashboard: http://gb10-ip:9999

The `Dockerfile` adds a patch to the hindsight integration in hermes. It allows you to specify the `bank_id` when using the `hindsight_*` tools to use multiple banks.

## iDRAC MCP

```
cd ~/git
git clone https://github.com/TrevorSquillario/idrac-redfish-mcp.git
```

Update docker volume mapping in hermes `compose.yaml`
```
volumes:
 - ../idrac-redfish-mcp:/mcp/idrac-redfish-mcp:ro
```

Add the `mcp_servers:` config in `config.yaml`. Remember the envs are read from `.env` mapped to `/opt/data/.hermes/.env` in the container, aka `~/.hermes/.env`
```
mcp_servers:
  idrac:
    command: "/opt/venv/bin/python"
    args: ["/mcp/idrac-redfish-mcp/src/fastmcp-server.py"]
    env:
      IDRAC_USERNAME: ${IDRAC_USERNAME}
      IDRAC_PASSWORD: ${IDRAC_PASSWORD}
      IDRAC_SSL_VERIFY: ${IDRAC_SSL_VERIFY}
```

Restart the `hermes` container with `docker compose up -d hermes`. This will recreate the container, if it doesn't use `docker compose restart hermes`

## iDRAC Skills

```
cd ~/git
git clone https://github.com/TrevorSquillario/idrac-hermes-skills
```

Update docker volume mapping in hermes `compose.yaml`
```
volumes:
 - ../idrac-hermes-skills:/skills/idrac-hermes-skills:ro
```

hermes will automatically pickup the new/updated skill the next prompt

## OME MCP
Note:
*If you're GB10 can't resolve the FQDN of the OME server. You can update the hosts file in the `compose.yaml`
```
  hermes:
    extra_hosts:
      - "ome.example.com:192.168.0.160"
```

1. Set OME appliance hostname to a proper FQDN. Configure DNS servers for OME.
2. Add the MCP server to your hermes `config.yaml` in the `mcp_servers` section
```
mcp_servers:
  ome:
    url: https://ome.example.com/mcp
    auth: oauth
    ssl_verify: false
```
or
```
mcp_servers:
  ome:
    url: https://ome.example.com/mcp
    auth: oauth
    ssl_verify: /opt/certs/ome.example.com.crt
```
3. Start the hermes container and run `hermes mcp login ome`. You will be presented with an OAuth URL. Copy/paste or Ctrl+Click that link and login with your OME credentials. *You only have 40s so be quick with it!*
4. The browser will redirect to a 127.0.0.1 address. Copy the full URL. (If that doesn't work just copy from the `?code=*` to the end. Ex: `?code=zeqfBfyA_tXc-TNf7RN-WZugyR7SqpslKn_KpSZG0hg&state=r1687_fXjyPwLj4IMxIbMcYcC7auqvqTjgzpffgx8Wg`)
5. Paste that into the terminal window where you ran the `hermes mcp...` command.
6. On Success it should say `Authenticated — 26 tool(s) available`. 

### Troubleshooting

- `405 Method Not Allowed` means you're using `http` instead of `https`
- Do the FQDNs match for OME in the hermes `config.yaml`, the OME hostname, and the DNS record?
- Are you using `https` in your `config.yaml`?
- If you're not using 

## Camofox

https://github.com/jo-inc/camofox-browser.git

Uncomment the `CAMOFOX_URL=` line in your `.env`

```
git clone https://github.com/jo-inc/camofox-browser.git
cd camofox-browser
make build

# A new image will get build
docker image ls | grep camo

docker compose --profile camofox up -d
```