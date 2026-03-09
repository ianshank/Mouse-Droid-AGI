#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Jetson Nano Hardware Driver Verification & Setup
#
# Idempotent script that verifies GPIO, UART, CSI camera, and I2C
# peripherals are accessible and properly configured.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
error() { echo "[ERROR] $*"; }
ok()    { echo "[OK]    $*"; }

section() {
    echo ""
    echo "========================================"
    echo "  $*"
    echo "========================================"
}

# ---------------------------------------------------------------------------
# 1. GPIO access
# ---------------------------------------------------------------------------

section "GPIO Access Verification"

GPIO_CHIPS=(/dev/gpiochip*)
if [ ${#GPIO_CHIPS[@]} -eq 0 ] || [ ! -e "${GPIO_CHIPS[0]}" ]; then
    error "No /dev/gpiochip* devices found."
    exit 1
fi

for chip in "${GPIO_CHIPS[@]}"; do
    if [ -r "$chip" ] && [ -w "$chip" ]; then
        ok "GPIO chip $chip is readable and writable"
    else
        warn "GPIO chip $chip has restricted permissions"
    fi
done

# Udev rule for GPIO access (idempotent)
UDEV_GPIO_RULE="/etc/udev/rules.d/99-gpio.rules"
GPIO_RULE_CONTENT='SUBSYSTEM=="gpio", KERNEL=="gpiochip*", MODE="0660", GROUP="gpio"'

if [ -f "$UDEV_GPIO_RULE" ]; then
    ok "GPIO udev rule already exists at $UDEV_GPIO_RULE"
else
    info "Adding GPIO udev rule..."
    echo "$GPIO_RULE_CONTENT" | sudo tee "$UDEV_GPIO_RULE" > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ok "GPIO udev rule installed"
fi

# Ensure gpio group exists and current user is a member
if getent group gpio > /dev/null 2>&1; then
    ok "gpio group exists"
else
    info "Creating gpio group..."
    sudo groupadd gpio
    ok "gpio group created"
fi

if id -nG "$USER" | grep -qw gpio; then
    ok "User $USER is in gpio group"
else
    info "Adding $USER to gpio group..."
    sudo usermod -aG gpio "$USER"
    ok "Added $USER to gpio group (re-login required)"
fi

# Test HC-SR04 pins (GPIO 23 = trigger, GPIO 24 = echo)
info "Testing GPIO 23 (trigger) and GPIO 24 (echo) for HC-SR04..."
python3 -c "
import sys
try:
    import gpiod
    chip = gpiod.Chip('gpiochip0')
    line23 = chip.get_line(23)
    line24 = chip.get_line(24)
    print('[OK]    GPIO 23 and 24 are accessible via gpiod')
except ImportError:
    print('[WARN]  python3-gpiod not installed — skipping GPIO pin test')
except Exception as e:
    print(f'[WARN]  GPIO pin test failed: {e}')
" || warn "GPIO pin test script failed"

# ---------------------------------------------------------------------------
# 2. UART / Serial verification
# ---------------------------------------------------------------------------

section "UART / Serial Verification"

SERIAL_PORT="/dev/ttyUSB0"

if [ -e "$SERIAL_PORT" ]; then
    ok "Serial port $SERIAL_PORT exists"
else
    warn "Serial port $SERIAL_PORT not found — connect ESP32 and retry"
fi

# Ensure user is in dialout group
if id -nG "$USER" | grep -qw dialout; then
    ok "User $USER is in dialout group"
else
    info "Adding $USER to dialout group..."
    sudo usermod -aG dialout "$USER"
    ok "Added $USER to dialout group (re-login required)"
fi

# Disable conflicting serial console (idempotent)
SERIAL_GETTY="serial-getty@ttyS0.service"
if systemctl is-active --quiet "$SERIAL_GETTY" 2>/dev/null; then
    info "Stopping $SERIAL_GETTY..."
    sudo systemctl stop "$SERIAL_GETTY"
    sudo systemctl disable "$SERIAL_GETTY"
    ok "Disabled $SERIAL_GETTY"
else
    ok "$SERIAL_GETTY is already stopped or not present"
fi

# Test serial port access
if [ -e "$SERIAL_PORT" ]; then
    python3 -c "
import serial
try:
    s = serial.Serial('$SERIAL_PORT', 1000000, timeout=0.5)
    s.close()
    print('[OK]    Serial port $SERIAL_PORT opened successfully at 1M baud')
except ImportError:
    print('[WARN]  pyserial not installed — skipping serial test')
except Exception as e:
    print(f'[WARN]  Serial port test failed: {e}')
" || warn "Serial port test script failed"
fi

# ---------------------------------------------------------------------------
# 3. CSI Camera verification
# ---------------------------------------------------------------------------

section "CSI Camera Verification"

if command -v v4l2-ctl &> /dev/null; then
    info "Detected video devices:"
    v4l2-ctl --list-devices 2>/dev/null || warn "No video devices detected"
else
    warn "v4l2-utils not installed — installing..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq v4l2-utils
    info "Detected video devices:"
    v4l2-ctl --list-devices 2>/dev/null || warn "No video devices detected"
fi

# Install camera dependencies (idempotent — apt handles already-installed)
info "Installing camera dependencies..."
sudo apt-get install -y -qq python3-libcamera python3-kms++ 2>/dev/null \
    || warn "Some camera packages unavailable (may not be needed on Jetson)"

# Test CSI camera access
python3 -c "
import sys

# Try jetson_utils first
try:
    import jetson_utils
    cam = jetson_utils.videoSource('csi://0', argv=['--input-width=640', '--input-height=480'])
    print('[OK]    jetson_utils CSI camera accessible')
    del cam
    sys.exit(0)
except ImportError:
    print('[WARN]  jetson_utils not available')
except Exception as e:
    print(f'[WARN]  jetson_utils camera test failed: {e}')

# Try OpenCV GStreamer fallback
try:
    import cv2
    gst = (
        'nvarguscamerasrc ! '
        'video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! '
        'nvvidconv ! video/x-raw,format=BGRx ! '
        'videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print('[OK]    OpenCV GStreamer CSI camera accessible')
        cap.release()
    else:
        print('[WARN]  OpenCV GStreamer pipeline could not open CSI camera')
except ImportError:
    print('[WARN]  OpenCV not available — skipping GStreamer test')
except Exception as e:
    print(f'[WARN]  OpenCV GStreamer test failed: {e}')
" || warn "CSI camera test script failed"

# ---------------------------------------------------------------------------
# 4. I2C verification
# ---------------------------------------------------------------------------

section "I2C Verification"

if command -v i2cdetect &> /dev/null; then
    ok "i2c-tools already installed"
else
    info "Installing i2c-tools..."
    sudo apt-get install -y -qq i2c-tools
    ok "i2c-tools installed"
fi

info "Scanning I2C bus 1..."
i2cdetect -y 1 2>/dev/null || warn "I2C bus 1 scan failed (bus may not exist)"

# Ensure user is in i2c group
if getent group i2c > /dev/null 2>&1; then
    if id -nG "$USER" | grep -qw i2c; then
        ok "User $USER is in i2c group"
    else
        info "Adding $USER to i2c group..."
        sudo usermod -aG i2c "$USER"
        ok "Added $USER to i2c group (re-login required)"
    fi
else
    warn "i2c group does not exist — I2C access may require root"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section "Setup Complete"

info "All checks finished. Review any [WARN] messages above."
info "If group memberships were changed, log out and log back in."
