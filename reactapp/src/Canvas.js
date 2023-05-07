"use client"

import React, { useRef, useEffect, useState } from 'react';

const Canvas = () => {
  const canvasRef = useRef(null);
  const A = useState({ x: 10, y: 10});
  const B = useState({ x: 20, y: 10});
  const C = useState({ x: 30, y: 10});
  const D = useState({ x: 40, y: 10});
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.fillStyle = 'rgba(0, 0, 255, 0.5)';
    context.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const handleMouseMove = (event) => {
    const { offsetX, offsetY } = event.nativeEvent;
    setMousePosition({ x: offsetX, y: offsetY });
  };

  const handleMouseOut = () => {
    setMousePosition({ x: 0, y: 0 });
  };


  return (
    <>
      <div style={{display:'flex', flexDirection:'column', gap: '1em', alignItems: 'center'}}>
       <Point point={A} name={"Point A:"} />
       <Point point={B} name={"Point B:"} />
       <Point point={C} name={"Point C:"} />
       <Point point={D} name={"Point D:"} />
      </div>
      <canvas
        ref={canvasRef}
        width={'400px'}
        height={'400px'}
        onMouseMove={handleMouseMove}
        onMouseOut={handleMouseOut}
      />
      <p>
        Mouse coordinates: ({mousePosition.x}, {mousePosition.y})
      </p>
      <p>
        Interpolated points between the 4 corners:
        <pre>
          {/* {JSON.stringify(calcPoint(mousePosition.x, mousePosition.y, 400, 400, A, B, C, D), null, 2)} */}
          {JSON.stringify(calcPoint(mousePosition.x, mousePosition.y, 400, 400, A, B, C, D))}
        </pre>
      </p>
    </>
  );
};


  // interpolate mouseposition based on % of canvas width/height, and 
  // based on X, Y of A, B, C, D
  // return the interpolated point
  // https://stackoverflow.com/questions/13433424/how-to-interpolate-points-in-a-2d-space
  function lerp(start, end, amt) {
    return (1 - amt) * start + amt * end;
  }
  
  function calcPoint(mouseX, mouseY, canvasWidth, canvasHeight, A, B, C, D) {
    // console.log("mouseX, mouseY, canvasWidth, canvasHeight, A, B, C, D", mouseX, mouseY, canvasWidth, canvasHeight, A, B, C, D)
    const aXY = A[0];
    const bXY = B[0];
    const cXY = C[0];
    const dXY = D[0];
  
    // Normalize the mouse position based on canvas width and height
    const normalizedMouseX = mouseX / canvasWidth;
    const normalizedMouseY = mouseY / canvasHeight;
  
    // Interpolate points between A and B, and C and D for X-axis
    const abX = lerp(aXY.x, bXY.x, normalizedMouseX);
    const cdX = lerp(cXY.x, dXY.x, normalizedMouseX);
  
    // Interpolate points between A and B, and C and D for Y-axis
    const abY = lerp(aXY.y, bXY.y, normalizedMouseX);
    const cdY = lerp(cXY.y, dXY.y, normalizedMouseX);
  
    // Interpolate the final point between the above results based on mouseY
    const interpolatedX = lerp(abX, cdX, normalizedMouseY);
    const interpolatedY = lerp(abY, cdY, normalizedMouseY);
  
    // console.log({interpolatedX})
    // Return the interpolated point
    return [Math.round(interpolatedX * 100)/100, Math.round(interpolatedY * 100)/100];
  }
  

function Point({point, name}) {
  const [data, setData] = point
  return (
    <div style={{display: 'inline-flex', gap: '1em'}}>
      <span>{name}</span>
      X: <input className="text-black" type="number" value={data.x} onChange={(e) => setData({...data, x: Math.round(e.target.value * 100) / 100})} />
      Y: <input className="text-black" type="number" value={data.y} onChange={(e) => setData({...data, y: Math.round(e.target.value * 100) / 100})} />
    </div>
  )
}

export default Canvas;
