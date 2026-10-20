#!/bin/bash
set -e

# Fix ownership of npm-installed files that may have been added after the build
# (e.g. via `pi install` which runs npm under the hood)
NPM_DIR="${PI_HOME}/.pi/agent/npm/node_modules"
if [ -d "$NPM_DIR" ]; then
    chown -R pi:pi "$NPM_DIR"
fi

# Drop privileges to the pi user and execute the CMD
exec gosu pi "$@"
