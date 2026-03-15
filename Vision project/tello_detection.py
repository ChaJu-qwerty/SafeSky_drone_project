"""
=============================================================
 Tello Drone — Detección de Drones y Personas en Tiempo Real
=============================================================
Requisitos:
    pip install djitellopy ultralytics opencv-python

Conexión:
    1. Enciende el Tello
    2. Conéctate al WiFi del Tello (TELLO-XXXXXX)
    3. Ejecuta este script

Controles — haz clic en la ventana de video para activar teclas:
    T       → Despegar
    L       → Aterrizar
    ESC     → Aterrizar de emergencia y salir
    W / S   → Adelante / Atrás
    A / D   → Izquierda / Derecha
    Q / E   → Rotar izquierda / Rotar derecha
    R / F   → Subir / Bajar
    I       → Mostrar/ocultar telemetría
=============================================================
"""

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from djitellopy import Tello

# ============================================================
#   CONFIGURACIÓN
# ============================================================
MODEL_PATH   = Path(r'C:\Users\SOPORTE\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\TEC\SEMESTRE 6\Diseño y desarrollo de robots\PROYECTO\Vision project\drone_project\training\runs\drone_person_yolov8m\weights\best.pt')
CONF_THRESH = 0.35
IOU_THRESH  = 0.45
SPEED       = 30        # cm por movimiento
ROTATE_DEG  = 30        # grados por rotación
WIN_W       = 960
WIN_H       = 720
# ============================================================

CLASS_NAMES  = ['drone', 'person']
CLASS_COLORS = {0: (0, 255, 100), 1: (60, 130, 255)}


def draw_frame(frame, results, tello, show_telem, fps):
    h, w = frame.shape[:2]
    counts = {0: 0, 1: 0}

    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf   = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color  = CLASS_COLORS.get(cls_id, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Esquinas decorativas
        clen = min(15, (x2-x1)//4, (y2-y1)//4)
        for (cx, cy) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            dx = clen if cx == x1 else -clen
            dy = clen if cy == y1 else -clen
            cv2.line(frame, (cx,cy), (cx+dx,cy), color, 3)
            cv2.line(frame, (cx,cy), (cx,cy+dy), color, 3)

        label = f'{CLASS_NAMES[cls_id]} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        counts[cls_id] += 1

    # HUD contadores
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (220,85), (20,20,20), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f'Drones : {counts[0]}', (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLASS_COLORS[0], 2)
    cv2.putText(frame, f'Persons: {counts[1]}', (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLASS_COLORS[1], 2)
    cv2.putText(frame, f'FPS: {fps}', (8, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    # Alertas
    if counts[0] > 0:
        cv2.putText(frame, '! DRONE DETECTED', (w//2-130, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,220), 3)
    if counts[1] > 0:
        cv2.putText(frame, '! PERSON NEARBY', (w//2-120, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,165,255), 3)

    # Telemetría
    if show_telem:
        try:
            lines = [
                f"Bat : {tello.get_battery()}%",
                f"Alt : {tello.get_height()} cm",
                f"Temp: {tello.get_highest_temperature()} C",
            ]
            bw, bh = 155, len(lines)*24+10
            ov2 = frame.copy()
            cv2.rectangle(ov2, (w-bw-5,5), (w-5,bh), (20,20,20), -1)
            cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (w-bw, 24+i*24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
        except Exception:
            pass

    # Guía de teclas
    guide = 'T:Despegar  L:Aterrizar  WASD:Mover  QE:Rotar  RF:SubirBajar  ESC:Salir'
    cv2.putText(frame, guide, (8, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    return frame


def main():
    # ── Cargar modelo ──
    print(f'Cargando modelo: {MODEL_PATH}')
    if not MODEL_PATH.exists():
        print(f'Modelo no encontrado: {MODEL_PATH}')
        return
    model = YOLO(str(MODEL_PATH))
    print('Modelo cargado')

    # ── Conectar Tello ──
    tello = Tello()
    tello.connect()
    print(f'Tello conectado | Batería: {tello.get_battery()}%')
    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(1)

    flying     = False
    show_telem = True
    fps        = 0
    fps_count  = 0
    fps_time   = time.time()

    # Crear ventana OpenCV — las teclas se capturan aquí
    cv2.namedWindow('Tello Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tello Detection', WIN_W, WIN_H)
    print('\n🎮 Haz clic en la ventana de video y usa las teclas.')
    print('   T → Despegar | L → Aterrizar | ESC → Salir\n')

    try:
        while True:
            frame = frame_reader.frame
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (WIN_W, WIN_H))

            # Inferencia
            results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)

            # FPS
            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps       = fps_count
                fps_count = 0
                fps_time  = time.time()

            frame = draw_frame(frame, results, tello, show_telem, fps)
            cv2.imshow('Tello Detection', frame)

            # ── Captura de teclas con waitKey ──
            # 1ms de espera — suficiente para no bloquear el video
            key = cv2.waitKey(1) & 0xFF

            if key == ord('t') and not flying:
                print('Despegando...')
                tello.takeoff()
                flying = True

            elif key == ord('l') and flying:
                print('Aterrizando...')
                tello.land()
                flying = False

            elif key == 27:  # ESC
                if flying:
                    print('Aterrizaje de emergencia')
                    tello.land()
                    flying = False
                break

            elif flying:
                if   key == ord('w'): tello.move_forward(SPEED)
                elif key == ord('s'): tello.move_back(SPEED)
                elif key == ord('a'): tello.move_left(SPEED)
                elif key == ord('d'): tello.move_right(SPEED)
                elif key == ord('r'): tello.move_up(SPEED)
                elif key == ord('f'): tello.move_down(SPEED)
                elif key == ord('q'): tello.rotate_counter_clockwise(ROTATE_DEG)
                elif key == ord('e'): tello.rotate_clockwise(ROTATE_DEG)

            if key == ord('i'):
                show_telem = not show_telem

    except KeyboardInterrupt:
        print('\nInterrumpido por el usuario')

    finally:
        if flying:
            print('Aterrizando...')
            tello.land()
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()
        print('Cerrado correctamente')


if __name__ == '__main__':
    main()