#!/bin/bash

# Control script for OSM Multi-Process Components
# Author: Antigravity AI
# Usage: ./control.sh [start|stop|status]

# Resolve script directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOGS_DIR="${SCRIPT_DIR}/logs"
PIDS_DIR="${SCRIPT_DIR}/pids"

mkdir -p "${LOGS_DIR}"
mkdir -p "${PIDS_DIR}"

# Array of services to manage
# Format: "service_name:config_file"
SERVICES=(
    "osm_camera:osm_camera.conf"
    "osm_can:osm_can.conf"
    "osm_process:osm_process.conf"
)

# Color variables for pretty console logs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

start_services() {
    echo -e "${BLUE}=== Starting OSM Multi-Process Components ===${NC}"
    
    # Clean leftover Unix domain socket files to prevent 'Address already in use' errors
    rm -f /tmp/flame.ipc /tmp/osm_camera.ipc /tmp/osm_can.ipc /tmp/osm_process.ipc /tmp/image_stream_1.ipc /tmp/image_stream_2.ipc
    
    # Check if any service is already running
    for svc_info in "${SERVICES[@]}"; do
        svc="${svc_info%%:*}"
        pid_file="${PIDS_DIR}/${svc}.pid"
        if [ -f "${pid_file}" ]; then
            pid=$(cat "${pid_file}")
            if ps -p "${pid}" > /dev/null 2>&1; then
                echo -e "${RED}[Error] Service ${svc} (PID: ${pid}) is already running.${NC}"
                exit 1
            fi
        fi
    done

    # 1. Start osm_camera
    echo -e "${YELLOW}Starting osm_camera (Solectrix grabber & camera monitor)...${NC}"
    nohup ./flame --config osm_camera.conf -v debug > "${LOGS_DIR}/osm_camera.log" 2>&1 &
    camera_pid=$!
    echo "${camera_pid}" > "${PIDS_DIR}/osm_camera.pid"
    echo -e "${GREEN}[Success] osm_camera started with PID ${camera_pid}${NC}"

    # 2. Start osm_can
    echo -e "${YELLOW}Starting osm_can (Kvaser CAN interface)...${NC}"
    nohup ./flame --config osm_can.conf -v debug > "${LOGS_DIR}/osm_can.log" 2>&1 &
    can_pid=$!
    echo "${can_pid}" > "${PIDS_DIR}/osm_can.pid"
    echo -e "${GREEN}[Success] osm_can started with PID ${can_pid}${NC}"

    # 3. Delay to let PCIe grabber and I2C initialization complete before GPU inference starts
    echo -e "${BLUE}Waiting 2 seconds for hardware interface setup to settle...${NC}"
    sleep 2

    # 4. Start osm_process
    echo -e "${YELLOW}Starting osm_process (Monolithic inference / YOLO11-Face)...${NC}"
    nohup ./flame --config osm_process.conf -v debug > "${LOGS_DIR}/osm_process.log" 2>&1 &
    process_pid=$!
    echo "${process_pid}" > "${PIDS_DIR}/osm_process.pid"
    echo -e "${GREEN}[Success] osm_process started with PID ${process_pid}${NC}"

    echo -e "${GREEN}=== All services launched. Logs are available in bin/x86_64/logs/ ===${NC}"
}

stop_services() {
    echo -e "${BLUE}=== Stopping OSM Multi-Process Components ===${NC}"
    
    # Clean socket files
    rm -f /tmp/flame.ipc /tmp/osm_camera.ipc /tmp/osm_can.ipc /tmp/osm_process.ipc /tmp/image_stream_1.ipc /tmp/image_stream_2.ipc
    
    # Process services in reverse order for clean teardown
    # 1. Monolithic inference
    # 2. CAN interface
    # 3. Camera grabber
    REVERSE_SERVICES=(
        "osm_process:osm_process.conf"
        "osm_can:osm_can.conf"
        "osm_camera:osm_camera.conf"
    )

    for svc_info in "${REVERSE_SERVICES[@]}"; do
        svc="${svc_info%%:*}"
        pid_file="${PIDS_DIR}/${svc}.pid"
        if [ -f "${pid_file}" ]; then
            pid=$(cat "${pid_file}")
            if ps -p "${pid}" > /dev/null 2>&1; then
                echo -e "${YELLOW}Stopping ${svc} (PID: ${pid})...${NC}"
                # Send SIGTERM (15) for graceful cleanup
                sudo kill -15 "${pid}" > /dev/null 2>&1
                
                # Wait for up to 5 seconds for the process to exit
                for i in {1..5}; do
                    if ! ps -p "${pid}" > /dev/null 2>&1; then
                        break
                    fi
                    sleep 1
                done
                
                # Force kill if still running
                if ps -p "${pid}" > /dev/null 2>&1; then
                    echo -e "${RED}${svc} did not exit gracefully, sending SIGKILL...${NC}"
                    sudo kill -9 "${pid}" > /dev/null 2>&1
                fi
                echo -e "${GREEN}[Stopped] ${svc}${NC}"
            else
                echo -e "${YELLOW}Service ${svc} (PID: ${pid}) is not running.${NC}"
            fi
            rm -f "${pid_file}"
        else
            echo -e "${YELLOW}PID file not found for ${svc}, trying pkill...${NC}"
            config_file="${svc_info##*:}"
            sudo pkill -f "flame --config ${config_file}" > /dev/null 2>&1
        fi
    done
    
    echo -e "${GREEN}=== All services stopped. ===${NC}"
}

show_status() {
    echo -e "${BLUE}=== OSM Multi-Process Status ===${NC}"
    for svc_info in "${SERVICES[@]}"; do
        svc="${svc_info%%:*}"
        pid_file="${PIDS_DIR}/${svc}.pid"
        if [ -f "${pid_file}" ]; then
            pid=$(cat "${pid_file}")
            if ps -p "${pid}" > /dev/null 2>&1; then
                echo -e "Service: ${GREEN}%-15s${NC} | Status: ${GREEN}RUNNING${NC} (PID: ${pid})" "${svc}"
            else
                echo -e "Service: ${RED}%-15s${NC} | Status: ${RED}DEAD${NC} (PID from file: ${pid} - process not found)" "${svc}"
            fi
        else
            # Try searching the process table
            config_file="${svc_info##*:}"
            pid=$(pgrep -f "flame --config ${config_file}")
            if [ -n "${pid}" ]; then
                echo -e "Service: ${YELLOW}%-15s${NC} | Status: ${GREEN}RUNNING${NC} (PID: ${pid} - no PID file)" "${svc}"
            else
                echo -e "Service: ${RED}%-15s${NC} | Status: ${RED}STOPPED${NC}" "${svc}"
            fi
        fi
    done
}

case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 1
        ;;
esac

exit 0
