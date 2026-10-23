#!/bin/sh
# TokenSaver: install once, then Claude Code saves tokens automatically.
#
#   git clone https://github.com/lukeprofits/TokenSaver ~/.tokensaver/src && ~/.tokensaver/src/install.sh
#
# Checks what it needs, shows the full plan for this machine and asks once.
# Pass --yes to apply without asking, --dry-run to only show the plan.
# Undo everything with: ~/.tokensaver/src/tokensaver-cli uninstall
set -eu
here="$(cd "$(dirname "$0")" && pwd)"

py=""
for c in python3 python3.13 python3.12 python3.11 python3.10 python3.9; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
        py="$(command -v "$c")"
        break
    fi
done
if [ -z "$py" ]; then
    echo "TokenSaver needs Python 3.9 or newer (python3). Install it, then run this again." >&2
    exit 1
fi
if ! command -v claude >/dev/null 2>&1; then
    echo "Claude Code (\`claude\`) is not on PATH. Install it first: https://code.claude.com" >&2
    exit 1
fi
case "$(uname -s)" in
    Darwin|Linux) ;;
    *) echo "TokenSaver supports macOS and Linux." >&2; exit 1 ;;
esac
if [ "$here" != "$HOME/.tokensaver/src" ]; then
    echo "note: installing from $here. Keep it there: the service and settings point at it."
    echo "      (the usual place is ~/.tokensaver/src)"
    echo
fi
exec "$py" "$here/tokensaver-cli" setup "$@"
