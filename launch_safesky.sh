#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  launch_safesky.sh — Lanzador automático SafeSky Gazebo
#  Proyecto: SafeSky — Simulación de dron en Gazebo con Hybrid A* + ORCA + PX4
#  MR3001B · Tecnológico de Monterrey 2026
# ═══════════════════════════════════════════════════════════════════════════════
#
#  USO:
#    chmod +x launch_safesky.sh
#    ./launch_safesky.sh
#
#  REQUISITOS PREVIOS (ver README para instalación completa):
#    - Ubuntu 24.04
#    - ROS2 Jazzy instalado
#    - PX4-Autopilot compilado en ~/PX4-Autopilot
#    - MicroXRCEAgent instalado
#    - ws_px4 compilado con safesky_mission
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# ── Colores para output ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Rutas (ajustar si es necesario) ───────────────────────────────────────────
PX4_DIR="$HOME/PX4-Autopilot"
WS_DIR="$HOME/ws_px4"
WORLD="safesky_world"

# ── Banner ─────────────────────────────────────────────────────────────────────
echo -e "${CYAN}"
echo "  ███████╗ █████╗ ███████╗███████╗███████╗██╗  ██╗██╗   ██╗"
echo "  ██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║ ██╔╝╚██╗ ██╔╝"
echo "  ███████╗███████║█████╗  █████╗  ███████╗█████╔╝  ╚████╔╝ "
echo "  ╚════██║██╔══██║██╔══╝  ██╔══╝  ╚════██║██╔═██╗   ╚██╔╝  "
echo "  ███████║██║  ██║██║     ███████╗███████║██║  ██╗   ██║   "
echo "  ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝   ╚═╝   "
echo -e "${NC}"
echo -e "${BLUE}  SafeSky — Simulación Gazebo | Hybrid A* + ORCA + PX4${NC}"
echo -e "${BLUE}  MR3001B · Tecnológico de Monterrey 2026${NC}"
echo ""

# ── Verificar dependencias ─────────────────────────────────────────────────────
echo -e "${YELLOW}[1/5] Verificando dependencias...${NC}"

if [ ! -d "$PX4_DIR" ]; then
    echo -e "${RED}ERROR: PX4-Autopilot no encontrado en $PX4_DIR${NC}"
    echo "  Instala PX4 siguiendo el README antes de continuar."
    exit 1
fi

if [ ! -d "$WS_DIR/install" ]; then
    echo -e "${RED}ERROR: workspace ROS2 no compilado en $WS_DIR${NC}"
    echo "  Ejecuta: cd $WS_DIR && colcon build"
    exit 1
fi

if ! command -v MicroXRCEAgent &> /dev/null; then
    echo -e "${RED}ERROR: MicroXRCEAgent no encontrado${NC}"
    echo "  Instala el agente siguiendo el README."
    exit 1
fi

if [ ! -f "$PX4_DIR/Tools/simulation/gz/worlds/${WORLD}.sdf" ]; then
    echo -e "${RED}ERROR: Mundo $WORLD no encontrado${NC}"
    echo "  Ejecuta el setup del mundo desde el README."
    exit 1
fi

echo -e "${GREEN}  ✓ Todas las dependencias encontradas${NC}"

# ── Limpiar procesos anteriores ────────────────────────────────────────────────
echo -e "${YELLOW}[2/5] Limpiando procesos anteriores...${NC}"
pkill -f px4 2>/dev/null || true
pkill -f gz 2>/dev/null || true
pkill -f MicroXRCEAgent 2>/dev/null || true
pkill -f offboard_node 2>/dev/null || true
pkill -f viz_node 2>/dev/null || true
sleep 2
echo -e "${GREEN}  ✓ Procesos anteriores terminados${NC}"

# ── Source ROS2 y workspace ────────────────────────────────────────────────────
echo -e "${YELLOW}[3/5] Activando ROS2 y workspace...${NC}"
source /opt/ros/jazzy/setup.bash
source "$WS_DIR/install/setup.bash"
echo -e "${GREEN}  ✓ ROS2 Jazzy + workspace activos${NC}"

# ── Lanzar en terminales separadas usando gnome-terminal o xterm ───────────────
echo -e "${YELLOW}[4/5] Lanzando componentes...${NC}"

# Detectar terminal disponible
if command -v gnome-terminal &> /dev/null; then
    TERM_CMD="gnome-terminal --"
elif command -v xterm &> /dev/null; then
    TERM_CMD="xterm -e"
else
    # Fallback: usar tmux si está disponible
    if command -v tmux &> /dev/null; then
        USE_TMUX=true
    else
        echo -e "${RED}ERROR: No se encontró gnome-terminal, xterm ni tmux${NC}"
        echo "  Instala uno: sudo apt install xterm"
        exit 1
    fi
fi

if [ "$USE_TMUX" = true ]; then
    # ── Modo tmux ───────────────────────────────────────────────────────────────
    echo -e "${CYAN}  Usando tmux — sesión: safesky${NC}"

    tmux kill-session -t safesky 2>/dev/null || true
    tmux new-session -d -s safesky -x 220 -y 50

    # Panel T1 — PX4 + Gazebo
    tmux rename-window -t safesky 'SafeSky'
    tmux send-keys -t safesky "cd $PX4_DIR && PX4_GZ_WORLD=$WORLD make px4_sitl gz_x500" Enter

    # Panel T2 — DDS Agent
    tmux split-window -t safesky -h
    tmux send-keys -t safesky "sleep 8 && MicroXRCEAgent udp4 -p 8888" Enter

    # Panel T3 — Misión
    tmux split-window -t safesky -v
    tmux send-keys -t safesky "sleep 15 && source /opt/ros/jazzy/setup.bash && source $WS_DIR/install/setup.bash && ros2 run safesky_mission offboard_node" Enter

    # Panel T4 — KPIs
    tmux select-pane -t safesky:0.0
    tmux split-window -t safesky -v
    tmux send-keys -t safesky "sleep 16 && source /opt/ros/jazzy/setup.bash && source $WS_DIR/install/setup.bash && ros2 run safesky_mission viz_node" Enter

    echo -e "${GREEN}  ✓ Sesión tmux 'safesky' creada${NC}"
    echo ""
    echo -e "${YELLOW}[5/5] Instrucciones finales:${NC}"
    echo -e "  Conecta a tmux con: ${CYAN}tmux attach -t safesky${NC}"
    echo ""
    echo -e "  Cuando veas el dron en Gazebo y 'pxh>' en T1, escribe:"
    echo -e "  ${CYAN}commander arm --force${NC}"
    echo ""
    echo -e "  Navegación tmux:"
    echo -e "    ${CYAN}Ctrl+B, flecha${NC}  → moverse entre paneles"
    echo -e "    ${CYAN}Ctrl+B, d${NC}       → desconectar (deja corriendo)"
    echo -e "    ${CYAN}Ctrl+B, &${NC}       → cerrar ventana"
    echo ""
    tmux attach -t safesky

else
    # ── Modo terminales separadas ───────────────────────────────────────────────

    # T1 — PX4 + Gazebo
    echo -e "  ${CYAN}→ Lanzando T1: PX4 + Gazebo${NC}"
    $TERM_CMD bash -c "
        echo '=== T1: PX4 + Gazebo ===';
        cd $PX4_DIR;
        PX4_GZ_WORLD=$WORLD make px4_sitl gz_x500;
        exec bash" &
    sleep 2

    # T2 — DDS Agent
    echo -e "  ${CYAN}→ Lanzando T2: MicroXRCEAgent${NC}"
    $TERM_CMD bash -c "
        echo '=== T2: MicroXRCE-DDS Agent ===';
        sleep 6;
        MicroXRCEAgent udp4 -p 8888;
        exec bash" &
    sleep 1

    # T3 — Nodo misión
    echo -e "  ${CYAN}→ Lanzando T3: Nodo de misión${NC}"
    $TERM_CMD bash -c "
        echo '=== T3: SafeSky Mission Node ===';
        sleep 14;
        source /opt/ros/jazzy/setup.bash;
        source $WS_DIR/install/setup.bash;
        ros2 run safesky_mission offboard_node;
        exec bash" &
    sleep 1

    # T4 — KPIs
    echo -e "  ${CYAN}→ Lanzando T4: KPIs y visualización${NC}"
    $TERM_CMD bash -c "
        echo '=== T4: SafeSky KPIs ===';
        sleep 15;
        source /opt/ros/jazzy/setup.bash;
        source $WS_DIR/install/setup.bash;
        ros2 run safesky_mission viz_node;
        exec bash" &

    echo -e "${GREEN}  ✓ 4 terminales lanzadas${NC}"
    echo ""
    echo -e "${YELLOW}[5/5] Instrucciones finales:${NC}"
    echo ""
    echo -e "  Espera ~15 segundos a que todo inicie."
    echo -e "  Cuando veas ${CYAN}pxh>${NC} en T1 y el dron en Gazebo, escribe en T1:"
    echo ""
    echo -e "    ${CYAN}commander arm --force${NC}"
    echo ""
    echo -e "  El dron despegará y volará de A (esfera verde) a B (esfera azul)"
    echo -e "  evitando los obstáculos rojos y naranja."
    echo ""
fi

echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  SafeSky lanzado correctamente ✅${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
