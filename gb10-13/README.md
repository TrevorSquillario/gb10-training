# Lesson 13: Hermes Agent

This is my very opinionated hermes setup:

- `curator.enabled=false`: I'm not a fan of hermes editing my skills. I run the show here Herm.
- `/skills` are mounted readonly
- `config.yaml` is mounted readonly
- `hindsight` for memory with `"auto_retain": false, "auto_recall": false` set to disable memory being automaticaly saved and recalled for every prompt
- `camofox` for web scraping

## Start
```
cd hermes
docker compose up -d
```

Add to your ~/.bashrc for a `hermes` command alias on your docker host
```
hermes() {
  docker exec -it hermes /opt/hermes/.venv/bin/hermes "$@"
}
```

Run the `hermes` command to launch the TUI or browse to the Hermes Dashboard at http://gb10-ip:9119

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

Config file: `hindsight.config.json`

Hindsight Dashboard: http://gb10-ip:9999

The `Dockerfile` adds a patch to the hindsight integration in hermes. It allows you to specify the `bank_id` when using the `hindsight_*` tools to use multiple banks.

## Home Assistant MCP Server

https://github.com/homeassistant-ai/ha-mcp
The Unofficial Home Assistant (ha) MCP server provides a wide range of tools for interacting with your smart home. It's considerably better than the built-in HA integration.

Here is a categorized list of the available tools:

```
System & Overview
ha_get_overview: Get a system-wide summary of entities, areas, and system status.
ha_get_system_health: Check the health of integrations, Zigbee/Z-Wave networks, and diagnostics.
ha_get_updates: Check for available updates (Core, Add-ons, Device Firmware, etc.).
ha_get_addon: Get detailed information about specific installed or available add-ons.
ha_manage_addon: Manage add-ons (configure settings, call APIs, or manage boot/update modes).

Entity & Device Management
ha_get_state: Get the current state and attributes of one or more entities.
ha_search_entities: Find entities by name, domain, or area.
ha_get_entity: Retrieve detailed registry information (area, icon, name, etc.) for an entity.
ha_set_entity: Update entity metadata (name, icon, area, visibility, device class, etc.).
ha_remove_entity: Permanently remove an entity from the registry.
ha_get_device: Get detailed information about a device.
ha_set_device: Update device properties (name, area, labels, or disable it).
ha_remove_device: Remove an orphaned device from the registry.
ha_get_integration: Get information about a configured integration.
ha_get_entity_exposure: Check which voice assistants (Alexa, Google, Assist) can see an entity.
ha_set_entity_exposure: Control voice assistant visibility for an entity.

Control & Execution
ha_call_service: The universal tool to control entities (lights, climate, etc.) and trigger automations.
ha_list_services: Discover all available services and their required parameters.
ha_get_camera_image: Retrieve a snapshot from a camera entity for visual analysis.
ha_get_operation_status: Check the real-time status of a service operation.

Configuration Management
ha_config_set_automation: Create or update automations (supports full config or Python transforms).
ha_config_remove_automation: Delete an automation.
ha_config_get_automation: Retrieve the configuration of an automation.
ha_config_get_automation_traces: Debug automations using execution traces.
ha_config_set_script: Create or update scripts.
ha_config_remove_script: Delete a script.
ha_config_get_script: Retrieve script configuration.
ha_config_set_scene: Create or update scenes.
ha_config_remove_scene: Delete a scene.
ha_config_get_scene: Retrieve scene configuration.
ha_config_set_dashboard: Create or update Lovelace dashboards.
ha_config_get_dashboard: Retrieve dashboard configuration or search for specific cards.
ha_config_set_dashboard_resource: Manage custom dashboard resources (JS, CSS, modules).
ha_config_remove_dashboard_resource: Remove a dashboard resource.
ha_config_set_category: Create or update organizational categories.
ha_config_get_category: List or retrieve category configurations.
ha_config_remove_category: Delete a category.
ha_config_set_label: Create or update labels for organization.
ha_config_get_label: List or retrieve label information.
ha_config_remove_label: Delete a label.

Helpers & Organizational Tools
ha_config_set_helper: Unified tool to create/update all helper types (input_boolean, template sensors, utility meters, etc.).
ha_config_remove_helpers_integrations: Remove a helper or an entire integration config entry.
ha_config_list_helpers: List all helpers of a specific type.
ha_config_set_group: Manage service-based entity groups.
ha_config_list_groups: List all existing entity groups.
ha_config_remove_group: Remove a service-based group.
ha_config_set_area_or_floor: Manage the building topology (areas and floors).
ha_list_floors_areas: Get a snapshot of floors and their assigned areas.
ha_remove_area_or_floor: Delete an area or a floor.
ha_config_set_zone: Create or update geographical zones.
ha_get_zone: Get details for a specific zone.
ha_remove_zone: Delete a zone.
ha_config_set_todo: Create or update todo list items.
ha_get_todo: List todo lists or retrieve items from a specific list.
ha_remove_todo_item: Delete an item from a todo list.

Advanced & Maintenance
ha_get_history: Retrieve historical state data or long-term statistics.
ha_get_logs: Access system, error, or add-on logs.
ha_get_blueprint: Retrieve details about automation or script blueprints.
ha_get_automation_traces: Debugging tool for automation execution flow.
ha_get_skill_guide: Access bundled best-practice documentation.
ha_manage_backup: Manage system snapshots or per-entity auto-backups.
ha_reload_core: Reload specific components (automations, scripts, etc.) without a full restart.
ha_restart: Perform a full system restart.
ha_manage_pipeline: Manage Assist voice pipelines.
ha_manage_energy_prefs: Configure the Energy Dashboard.
```