#!/usr/bin/env bash
# ============================================================================
# Поиск ссылок на ADR в коде.
#
#   docs/adr/find.sh 0002     # все маркеры ADR-0002 / docs/adr/0002-*
#   docs/adr/find.sh all      # по всем ADR из docs/adr/
#   docs/adr/find.sh --check  # взаимная проверка ADR <-> код
#
# Скрипт ищет два вида маркеров:
#   * локальный комментарий у функции:  // ADR-0002: docs/adr/0002-....md
#   * шапка модуля:                     // ADR: docs/adr/0002-....md
#
# Номера строк в ADR не фиксируются: актуальные места всегда печатает этот
# скрипт, поэтому сдвиг кода не ломает документацию.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ADR_DIR="$ROOT/docs/adr"
SEARCH_DIRS=(include src bindings training benchmarks tests tools)
EXTRA_FILES=(CMakeLists.txt Makefile)
GREP_INCLUDES=(
    --include='*.h' --include='*.hpp' --include='*.hxx'
    --include='*.c' --include='*.cc' --include='*.cpp'
    --include='*.py' --include='Makefile' --include='*.txt'
)

usage() {
    sed -n '3,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

targets() {
    local d f
    for d in "${SEARCH_DIRS[@]}"; do
        [[ -d "$ROOT/$d" ]] && printf '%s\n' "$d"
    done
    for f in "${EXTRA_FILES[@]}"; do
        [[ -f "$ROOT/$f" ]] && printf '%s\n' "$f"
    done
}

doc_ids() {
    local f
    for f in "$ADR_DIR"/[0-9][0-9][0-9][0-9]-*.md; do
        [[ -e "$f" ]] || continue
        basename "$f" | cut -c1-4
    done | sort -u
}

find_id() {
    local id="$1" rc=0
    local -a t=()
    mapfile -t t < <(targets)
    ( cd "$ROOT" && grep -rnE --binary-files=without-match \
        "${GREP_INCLUDES[@]}" \
        "(ADR-${id}\\b|docs/adr/${id}-)" "${t[@]}" ) || rc=$?
    # grep: 0 — найдено, 1 — не найдено; >=128 — SIGPIPE при `| head`.
    if (( rc > 1 && rc < 128 )); then
        echo "find.sh: ошибка grep для ADR-${id}" >&2
        return "$rc"
    fi
    return 0
}

code_ids() {
    local -a t=()
    mapfile -t t < <(targets)
    ( cd "$ROOT" && grep -rhoE --binary-files=without-match \
        "${GREP_INCLUDES[@]}" \
        "(ADR-[0-9]{4}|docs/adr/[0-9]{4}-)" "${t[@]}" ) \
        | grep -oE '[0-9]{4}' | sort -u
}

check() {
    local rc=0 id
    local -A in_code=()
    while IFS= read -r id; do
        [[ -n "$id" ]] && in_code["$id"]=1
    done < <(code_ids)

    while IFS= read -r id; do
        if [[ -n "${in_code[$id]:-}" ]]; then
            printf 'ok      ADR-%s: маркеры в коде найдены\n' "$id"
        else
            printf 'FAIL    ADR-%s: в коде нет ни одного маркера\n' "$id" >&2
            rc=1
        fi
    done < <(doc_ids)

    while IFS= read -r id; do
        [[ -z "$id" ]] && continue
        if ! compgen -G "$ADR_DIR/$id-*.md" >/dev/null; then
            printf 'FAIL    в коде есть ADR-%s, но нет docs/adr/%s-*.md\n' "$id" "$id" >&2
            rc=1
        fi
    done < <(printf '%s\n' "${!in_code[@]}" | sort -u)

    if (( rc == 0 )); then
        echo "проверка пройдена: все ADR упомянуты в коде"
    fi
    return "$rc"
}

main() {
    local arg="${1:-}"
    case "$arg" in
        ""|-h|--help|help) usage 0 ;;
        --check) check ;;
        all)
            local id
            while IFS= read -r id; do
                printf '### ADR-%s\n' "$id"
                find_id "$id"
                echo
            done < <(doc_ids)
            ;;
        [0-9][0-9][0-9][0-9])
            find_id "$arg"
            ;;
        *)
            echo "find.sh: ожидается NNNN, all или --check" >&2
            usage 1
            ;;
    esac
}

main "$@"
