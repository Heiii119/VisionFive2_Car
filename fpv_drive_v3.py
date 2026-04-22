#!/usr/bin/env python3
from flask import Flask, Response, request, jsonify
import subprocess
import socket
import time
import threading
import signal

# =========================
# Camera config
# =========================
DEVICE = "/dev/video4"
WIDTH = 320
HEIGHT = 240
FPS = 10
PORT = 6060

# =========================
# PCA9685 / PWM CONFIG
# =========================
PCA9685_ADDR = 0x40
PCA9685_FREQ = 60
I2C_BUS = 0

THROTTLE_CHANNEL = 0
STEERING_CHANNEL = 1

THROTTLE_STOPPED_TICKS = 370
THROTTLE_FORWARD_TICKS = 415
THROTTLE_REVERSE_TICKS = 305

STEERING_CENTER_TICKS = 380
STEERING_MIN_TICKS = 305
STEERING_MAX_TICKS = 480

STEP = 5
STEERING_STEP = 25

FAILSAFE_TIMEOUT_SEC = 0.35

# =========================
# Flask
# =========================
app = Flask(__name__)

# =========================
# HTML UI
# =========================
HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport"
content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"/>
<title>FPV Dual Joystick</title>
<style>
html,body{margin:0;padding:0;height:100%;background:#000;overflow:hidden;touch-action:none;}
.wrap{display:flex;height:100%;width:100%;}
.joyPanel{flex:1;display:flex;align-items:center;justify-content:center;background:#111;}
.videoPanel{flex:2;display:flex;align-items:center;justify-content:center;background:#000;}
.videoPanel img{width:100%;height:100%;object-fit:contain;}
.joystick{width:180px;height:180px;border-radius:50%;
background:rgba(255,255,255,0.08);
border:2px solid rgba(255,255,255,0.2);position:relative;}
.knob{width:70px;height:70px;border-radius:50%;
background:rgba(0,255,120,0.8);
position:absolute;left:50%;top:50%;
transform:translate(-50%,-50%);}
</style>
</head>
<body>
<div class="wrap">
<div class="joyPanel">
<div id="steerJoy" class="joystick"><div class="knob"></div></div>
</div>
<div class="videoPanel">
<img src="/mjpg">
</div>
<div class="joyPanel">
<div id="throttleJoy" class="joystick"><div class="knob"></div></div>
</div>
</div>
<script>
const state={up:false,down:false,left:false,right:false,center:false,brake:false};
let inFlight=false;
function send(){
if(inFlight)return;
inFlight=true;
fetch("/control",{method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify(state)})
.finally(()=>{inFlight=false;});
}
setInterval(send,50);

function setupJoystick(el,type){
const knob=el.querySelector(".knob");
const radius=el.clientWidth/2;
let active=false;
function update(x,y){
const rect=el.getBoundingClientRect();
const dx=x-rect.left-radius;
const dy=y-rect.top-radius;
const max=radius-35;
const dist=Math.sqrt(dx*dx+dy*dy);
let nx=dx,ny=dy;
if(dist>max){nx=dx/dist*max;ny=dy/dist*max;}
knob.style.left=(radius+nx)+"px";
knob.style.top=(radius+ny)+"px";
if(type==="steer"){
state.left=nx<-20;
state.right=nx>20;
state.center=Math.abs(nx)<=20;
}
if(type==="throttle"){
state.up=ny<-20;
state.down=ny>20;
state.brake=false;
}
}
function reset(){
knob.style.left="50%";
knob.style.top="50%";
knob.style.transform="translate(-50%,-50%)";
if(type==="steer"){
state.left=false;state.right=false;state.center=true;
}
if(type==="throttle"){
state.up=false;state.down=false;state.brake=true;
}
}
el.addEventListener("pointerdown",e=>{active=true;el.setPointerCapture(e.pointerId);update(e.clientX,e.clientY);});
el.addEventListener("pointermove",e=>{if(active)update(e.clientX,e.clientY);});
el.addEventListener("pointerup",()=>{active=false;reset();});
el.addEventListener("pointercancel",reset);
reset();
}
setupJoystick(document.getElementById("steerJoy"),"steer");
setupJoystick(document.getElementById("throttleJoy"),"throttle");
</script>
</body>
</html>
"""

# =========================
# Control State
# =========================
state_lock = threading.Lock()
control_state = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "center": False,
    "brake": False,
    "last_seen": 0.0,
}

# =========================
# Camera streaming
# =========================
_latest_lock = threading.Lock()
_latest_jpeg = None
_latest_seq = 0

def ffmpeg_jpeg_pipe():
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "video4linux2",
        "-framerate", str(FPS),
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-i", DEVICE,
        "-an",
        "-c:v", "mjpeg",
        "-q:v", "7",
        "-f", "image2pipe",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0)

def iter_jpegs(stream):
    buf=bytearray()
    while True:
        chunk=stream.read(4096)
        if not chunk:
            return
        buf.extend(chunk)
        while True:
            soi=buf.find(b"\xff\xd8")
            eoi=buf.find(b"\xff\xd9")
            if soi!=-1 and eoi!=-1:
                jpg=bytes(buf[soi:eoi+2])
                del buf[:eoi+2]
                yield jpg
            else:
                break

def camera_worker():
    global _latest_jpeg,_latest_seq
    while True:
        p=ffmpeg_jpeg_pipe()
        try:
            for jpg in iter_jpegs(p.stdout):
                with _latest_lock:
                    _latest_jpeg=jpg
                    _latest_seq+=1
        finally:
            try: p.kill()
            except: pass
        time.sleep(0.2)

def mjpeg_generator():
    boundary=b"--frame\r\n"
    last=-1
    while True:
        with _latest_lock:
            seq=_latest_seq
            jpg=_latest_jpeg
        if jpg is None or seq==last:
            time.sleep(0.01)
            continue
        last=seq
        yield (boundary+
               b"Content-Type: image/jpeg\r\n"+
               f"Content-Length: {len(jpg)}\r\n\r\n".encode()+
               jpg+b"\r\n")

# =========================
# PWM (Simple Direct Logic)
# =========================
from smbus2 import SMBus

bus = SMBus(I2C_BUS)

def clamp(v,a,b): return max(a,min(b,v))

def set_pwm(channel,value):
    value=int(clamp(value,0,4095))
    base=0x06+4*channel
    bus.write_byte_data(PCA9685_ADDR,base,value&0xFF)
    bus.write_byte_data(PCA9685_ADDR,base+1,(value>>8)&0xFF)
    bus.write_byte_data(PCA9685_ADDR,base+2,0)
    bus.write_byte_data(PCA9685_ADDR,base+3,0)

throttle=THROTTLE_STOPPED_TICKS
steering=STEERING_CENTER_TICKS

def control_loop():
    global throttle,steering
    while True:
        time.sleep(0.02)
        with state_lock:
            s=dict(control_state)
        if time.perf_counter()-s["last_seen"]>FAILSAFE_TIMEOUT_SEC:
            s["brake"]=True
        if s["brake"]:
            throttle=THROTTLE_STOPPED_TICKS
        elif s["up"]:
            throttle=clamp(throttle+STEP,THROTTLE_REVERSE_TICKS,THROTTLE_FORWARD_TICKS)
        elif s["down"]:
            throttle=clamp(throttle-STEP,THROTTLE_REVERSE_TICKS,THROTTLE_FORWARD_TICKS)
        if s["center"]:
            steering=STEERING_CENTER_TICKS
        elif s["left"]:
            steering=clamp(steering+STEERING_STEP,STEERING_MIN_TICKS,STEERING_MAX_TICKS)
        elif s["right"]:
            steering=clamp(steering-STEERING_STEP,STEERING_MIN_TICKS,STEERING_MAX_TICKS)
        set_pwm(THROTTLE_CHANNEL,throttle)
        set_pwm(STEERING_CHANNEL,steering)

# =========================
# Routes
# =========================
@app.get("/")
def index(): return HTML

@app.get("/mjpg")
def mjpg():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/control")
def control():
    data=request.get_json(force=True) or {}
    with state_lock:
        for k in ["up","down","left","right","center","brake"]:
            control_state[k]=bool(data.get(k,False))
        control_state["last_seen"]=time.perf_counter()
    return jsonify(ok=True)

# =========================
# Main
# =========================
if __name__=="__main__":
    threading.Thread(target=camera_worker,daemon=True).start()
    threading.Thread(target=control_loop,daemon=True).start()
    app.run(host="0.0.0.0",port=PORT,threaded=True,use_reloader=False)
