HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport"
content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"/>

<title>FPV Dual Joystick</title>

<style>
html, body{
margin:0;
padding:0;
height:100%;
background:#000;
overflow:hidden;
touch-action:none;
user-select:none;
-webkit-user-select:none;
-webkit-touch-callout:none;
}

*{
box-sizing:border-box;
-webkit-tap-highlight-color:transparent;
}

.wrap{
display:flex;
height:100%;
width:100%;
flex-direction:row;
}

.joyPanel{
flex:1;
display:flex;
align-items:center;
justify-content:center;
background:#111;
}

.videoPanel{
flex:2;
display:flex;
align-items:center;
justify-content:center;
background:#000;
}

.videoPanel img{
width:100%;
height:100%;
object-fit:contain;
}

.joystick{
width:180px;
height:180px;
border-radius:50%;
background:rgba(255,255,255,0.08);
border:2px solid rgba(255,255,255,0.2);
position:relative;
touch-action:none;
}

.knob{
width:70px;
height:70px;
border-radius:50%;
background:rgba(0,255,120,0.8);
position:absolute;
left:50%;
top:50%;
transform:translate(-50%,-50%);
}
</style>
</head>

<body>

<div class="wrap">

<div class="joyPanel">
<div id="steerJoy" class="joystick">
<div class="knob"></div>
</div>
</div>

<div class="videoPanel">
<img src="/mjpg">
</div>

<div class="joyPanel">
<div id="throttleJoy" class="joystick">
<div class="knob"></div>
</div>
</div>

</div>

<script>
document.addEventListener("touchmove",e=>e.preventDefault(),{passive:false});
document.addEventListener("gesturestart",e=>e.preventDefault());
document.addEventListener("selectstart",e=>e.preventDefault());

const state = {
up:false,
down:false,
left:false,
right:false,
center:false,
brake:false
};

let inFlight=false;

function send(){
if(inFlight) return;
inFlight=true;

fetch("/control",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify(state)
}).finally(()=>{inFlight=false;});
}

setInterval(send,50); // 20Hz heartbeat

function setupJoystick(el, type){

const knob = el.querySelector(".knob");
const radius = el.clientWidth/2;
let active=false;

function update(x,y){

const dx = x - el.getBoundingClientRect().left - radius;
const dy = y - el.getBoundingClientRect().top - radius;

const dist = Math.sqrt(dx*dx+dy*dy);
const max = radius-35;

let nx = dx;
let ny = dy;

if(dist > max){
nx = dx/dist * max;
ny = dy/dist * max;
}

knob.style.left = (radius + nx) + "px";
knob.style.top  = (radius + ny) + "px";

if(type==="steer"){
state.left  = nx < -20;
state.right = nx > 20;
state.center = Math.abs(nx) <= 20;
}

if(type==="throttle"){
state.up   = ny < -20;
state.down = ny > 20;
state.brake = false;
}

}

function reset(){
knob.style.left="50%";
knob.style.top="50%";
knob.style.transform="translate(-50%,-50%)";

if(type==="steer"){
state.left=false;
state.right=false;
state.center=true;
}

if(type==="throttle"){
state.up=false;
state.down=false;
state.brake=true;
}
}

el.addEventListener("pointerdown",e=>{
active=true;
el.setPointerCapture(e.pointerId);
update(e.clientX,e.clientY);
});

el.addEventListener("pointermove",e=>{
if(!active) return;
update(e.clientX,e.clientY);
});

el.addEventListener("pointerup",e=>{
active=false;
reset();
});

el.addEventListener("pointercancel",reset);

reset();
}

setupJoystick(document.getElementById("steerJoy"),"steer");
setupJoystick(document.getElementById("throttleJoy"),"throttle");

</script>

</body>
</html>
"""
