// Copyright (c) 2015, <your name>. All rights reserved. Use of this source code
// is governed by a BSD-style license that can be found in the LICENSE file.

import 'dart:html';
import 'package:vector_math/vector_math.dart';
import 'dart:web_gl' as webgl;
import 'dart:typed_data';
import 'dart:async';

class Renderer {
  Matrix4 _pMatrix;
  Matrix4 _mvMatrix;
  webgl.Program _shaderProgram;
  webgl.RenderingContext _gl;
  int _aVertexPosition;
  int _aVertexNormal;
  webgl.UniformLocation _uPMatrix;
  webgl.UniformLocation _uMVMatrix;
  webgl.UniformLocation _uNMatrix;
  bool _needUpdate;

  Point _startDrag;
  Matrix4 _rotation;
  Matrix3 _nMatrix;

  Vector3 rotateVector(Vector3 a, Vector3 b, Vector3 x) {
    return x.reflect(a.normalized()).reflect((a + b).normalized());
  }
  Matrix4 makeRotation(Vector3 a, Vector3 b) {
    Matrix3 rot;
    rot = new Matrix3.columns(rotateVector(a, b, new Vector3(1.0, 0.0, 0.0)),
        rotateVector(a, b, new Vector3(0.0, 1.0, 0.0)),
        rotateVector(a, b, new Vector3(0.0, 0.0, 1.0)));
    Matrix4 out = new Matrix4.identity();
    out.setRotation(rot);
    return out;
  }
  Matrix4 dragRotation(Point start, Point end, [double planeDistance = 1.0]) {
    Vector2 startDrag =
        new Vector2(start.x / _viewportWidth, 1.0 - start.y / _viewportHeight);
    Vector2 endDrag =
        new Vector2(end.x / _viewportWidth, 1.0 - end.y / _viewportHeight);

    startDrag -= new Vector2(0.5, 0.5);
    endDrag -= new Vector2(0.5, 0.5);
    Matrix4 rotation = makeRotation(
        new Vector3(startDrag.x, startDrag.y, -planeDistance),
        new Vector3(endDrag.x, endDrag.y, -planeDistance));

    return rotation;
  }
  Renderer(CanvasElement canvas) {
    _viewportWidth = canvas.width;
    _viewportHeight = canvas.height;

    double planeDistance = 1.0;
    _startDrag = null;
    _rotation = new Matrix4.identity();
    canvas.onMouseDown.listen((MouseEvent e) {
      _startDrag = e.client;
    });
    const int rotationPower = 3;

    canvas.onMouseMove.listen((MouseEvent e) {
      if (_startDrag != null) {
        Matrix4 rotation = dragRotation(_startDrag, e.client, planeDistance);
        for (int i = 0; i < rotationPower; i++) {
          _rotation = _rotation.multiply(rotation);
        }

        _startDrag = e.client;
        _needUpdate = true;
      }
    });

    canvas.onMouseUp.listen((MouseEvent e) {
      if (_startDrag != null) {
        Matrix4 rotation = dragRotation(_startDrag, e.client, planeDistance);
        for (int i = 0; i < rotationPower; i++) {
          _rotation = _rotation.multiply(rotation);
        }

        _startDrag = null;
        _needUpdate = true;
      }
    });
    canvas.onMouseOut.listen((MouseEvent e) {
      if (_startDrag != null) {
        Matrix4 rotation = dragRotation(_startDrag, e.client, planeDistance);
        for (int i = 0; i < rotationPower; i++) {
          _rotation = _rotation.multiply(rotation);
        }

        _startDrag = null;
        _needUpdate = true;
      }
    });

    _gl = canvas.getContext("experimental-webgl");

    _initShaders();
    _setupModel();

    _gl.clearColor(0.0, 0.0, 0.0, 1.0);
    _gl.enable(webgl.RenderingContext.DEPTH_TEST);
  }

  void _initShaders() {
    // vertex shader source code. uPosition is our variable that we'll
    // use to create animation
    String vsSource = """
    attribute vec3 vPosition;
    attribute vec3 vNormal;
    uniform mat4 uMVMatrix;
    uniform mat3 uNMatrix;
    uniform mat4 uPMatrix;
    varying vec3 fPosition;
    varying vec3 fNormal;
    varying vec3 fColor;
    void main(void) {
        fColor = vec3(1.0,1.0,1.0);
        fNormal = uNMatrix * vNormal;
        vec4 mvPos = uMVMatrix * vec4(vPosition,1.0);
        gl_Position = uPMatrix * mvPos;
        fPosition = gl_Position.xyz / gl_Position.w;
    }
    """;

    // fragment shader source code. uColor is our variable that we'll
    // use to animate color
    String fsSource = """
    precision mediump float;
    uniform sampler2D uSampler;
    varying vec3 fPosition;
    varying vec3 fNormal;
    varying vec3 fColor;
    void main(void) {
        float attenuation = 0.0;
        attenuation += max(0.0, dot(fNormal, -normalize(fPosition))); 
        gl_FragColor = vec4(fColor * attenuation,1.0);
    }
    """;

    // vertex shader compilation
    webgl.Shader vs = _gl.createShader(webgl.RenderingContext.VERTEX_SHADER);
    _gl.shaderSource(vs, vsSource);
    _gl.compileShader(vs);

    // fragment shader compilation
    webgl.Shader fs = _gl.createShader(webgl.RenderingContext.FRAGMENT_SHADER);
    _gl.shaderSource(fs, fsSource);
    _gl.compileShader(fs);

    // attach shaders to a WebGL program
    _shaderProgram = _gl.createProgram();
    _gl.attachShader(_shaderProgram, vs);
    _gl.attachShader(_shaderProgram, fs);
    _gl.linkProgram(_shaderProgram);
    _gl.useProgram(_shaderProgram);

    /**
     * Check if shaders were compiled properly. This is probably the most painful part
     * since there's no way to "debug" shader compilation
     */
    if (!_gl.getShaderParameter(vs, webgl.RenderingContext.COMPILE_STATUS)) {
      print(_gl.getShaderInfoLog(vs));
    }

    if (!_gl.getShaderParameter(fs, webgl.RenderingContext.COMPILE_STATUS)) {
      print(_gl.getShaderInfoLog(fs));
    }

    if (!_gl.getProgramParameter(
        _shaderProgram, webgl.RenderingContext.LINK_STATUS)) {
      print(_gl.getProgramInfoLog(_shaderProgram));
    }

    _vertexBuffer = _gl.createBuffer();
    _indexBuffer = _gl.createBuffer();

    _aVertexPosition = _gl.getAttribLocation(_shaderProgram, "vPosition");
    _gl.enableVertexAttribArray(_aVertexPosition);

    _aVertexNormal = _gl.getAttribLocation(_shaderProgram, "vNormal");
    _gl.enableVertexAttribArray(_aVertexNormal);

    _uPMatrix = _gl.getUniformLocation(_shaderProgram, "uPMatrix");
    _uMVMatrix = _gl.getUniformLocation(_shaderProgram, "uMVMatrix");
    _uNMatrix = _gl.getUniformLocation(_shaderProgram, "uNMatrix");
  }

  void _setMatrixUniforms() {
    Float32List tmpList = new Float32List(16);

    _pMatrix.copyIntoArray(tmpList);
    _gl.uniformMatrix4fv(_uPMatrix, false, tmpList);

    _mvMatrix.copyIntoArray(tmpList);
    _gl.uniformMatrix4fv(_uMVMatrix, false, tmpList);

    tmpList = new Float32List(9);
    Matrix3 nMatrix = new Matrix3.columns(
        _mvMatrix.row0.xyz, _mvMatrix.row1.xyz, _mvMatrix.row2.xyz);
    nMatrix.invert();
    nMatrix.copyIntoArray(tmpList);
    _gl.uniformMatrix3fv(_uNMatrix, false, tmpList);
  }

  static final int dimensions = 4;
  static final int stride = (dimensions * 4) * 2;
  static final int positionOffset = 0;
  static final int normalOffset = dimensions * 4;

  webgl.Buffer _vertexBuffer;
  webgl.Buffer _indexBuffer;
  
  void _addVertex(List<double> buffer, Vector3 p, Vector3 n) {
    List<double> tmp = new List<double>(4);
    p.copyIntoArray(tmp);
    tmp[3] = 1.0;
    buffer.addAll(tmp);
    n.copyIntoArray(tmp);
    buffer.addAll(tmp);
  }

  void _addTriangle(List<double> buffer, List<Vector3> a) {
    Vector3 normal;
    normal = (a[2] - a[0]).cross(a[1] - a[0]).normalize();
    for (int i = 0; i < 3; i++) {
      List<double> tmp = new List<double>(4);
      a[i].copyIntoArray(tmp);
      tmp[3] = 1.0;
      buffer.addAll(tmp);
      normal.copyIntoArray(tmp);
      buffer.addAll(tmp);
    }
  }
  int _vertexCount;
  void _setupModel() {
    List<double> buffer = new List<double>();

    /*List vertexes = [
      [[0.0, 1.0, 1.0], [0.0, 0.0, -1.0]],
      [[1.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
      [[1.0, 1.0, 1.0], [0.0, 0.0, -1.0]],
      
      [[1.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
      [[0.0, 1.0, 1.0], [0.0, 0.0, -1.0]],
      [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    ];*/
    
    List vertexes = [
      //
      [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
      [[0.0, 1.0, 1.0], [0.0, 0.0, -1.0]],
      [[1.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
      [[1.0, 1.0, 1.0], [0.0, 0.0, -1.0]],
      //
      [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
      [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
      [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    ];
    
    List<int> indexData = [
      //
      1,2,3,
      //
      2,1,0,
      //
      7,6,5,
      //
      4,5,6
      ];
    
    _vertexCount = indexData.length;

    for (int i = 0; i < vertexes.length; i++) {
      buffer.addAll(vertexes[i][0]);
      buffer.add(1.0);
      buffer.addAll(vertexes[i][1]);
      buffer.add(0.0);
    }

    /*
    // Front face.
    _addTriangle(buffer, [
      new Vector3(0.0, 1.0, 1.0),
      new Vector3(1.0, 0.0, 1.0),
      new Vector3(1.0, 1.0, 1.0)
    ]);
    _addTriangle(buffer, [
      new Vector3(1.0, 0.0, 1.0),
      new Vector3(0.0, 1.0, 1.0),
      new Vector3(0.0, 0.0, 1.0)
    ]);
    // Back face.
    _addTriangle(buffer, [
      new Vector3(1.0, 0.0, 0.0),
      new Vector3(0.0, 1.0, 0.0),
      new Vector3(1.0, 1.0, 0.0)
    ]);
    _addTriangle(buffer, [
      new Vector3(0.0, 1.0, 0.0),
      new Vector3(1.0, 0.0, 0.0),
      new Vector3(0.0, 0.0, 0.0)
    ]);
    // Left face.
    _addTriangle(buffer, [
      new Vector3(0.0, 1.0, 0.0),
      new Vector3(0.0, 0.0, 1.0),
      new Vector3(0.0, 1.0, 1.0)
    ]);
    _addTriangle(buffer, [
      new Vector3(0.0, 0.0, 1.0),
      new Vector3(0.0, 1.0, 0.0),
      new Vector3(0.0, 0.0, 0.0)
    ]);
    // Right face.
    _addTriangle(buffer, [
      new Vector3(1.0, 0.0, 1.0),
      new Vector3(1.0, 1.0, 0.0),
      new Vector3(1.0, 1.0, 1.0)
    ]);
    _addTriangle(buffer, [
      new Vector3(1.0, 1.0, 0.0),
      new Vector3(1.0, 0.0, 1.0),
      new Vector3(1.0, 0.0, 0.0)
    ]);
    // Bottom face.
    _addTriangle(buffer, [
      new Vector3(0.0, 0.0, 1.0),
      new Vector3(1.0, 0.0, 0.0),
      new Vector3(1.0, 0.0, 1.0)
    ]);
    _addTriangle(buffer, [
      new Vector3(1.0, 0.0, 0.0),
      new Vector3(0.0, 0.0, 1.0),
      new Vector3(0.0, 0.0, 0.0)
    ]);
    // Top face.
    _addTriangle(buffer, [
      new Vector3(1.0, 1.0, 0.0),
      new Vector3(0.0, 1.0, 1.0),
      new Vector3(1.0, 1.0, 1.0)
    ]);
    _addTriangle(buffer, [
      new Vector3(0.0, 1.0, 1.0),
      new Vector3(1.0, 1.0, 0.0),
      new Vector3(0.0, 1.0, 0.0)
    ]);
*/

    _gl.bindBuffer(webgl.RenderingContext.ARRAY_BUFFER, _vertexBuffer);
    _gl.vertexAttribPointer(_aVertexPosition, 3, webgl.RenderingContext.FLOAT,
        false, stride, positionOffset);
    _gl.vertexAttribPointer(_aVertexNormal, 3, webgl.RenderingContext.FLOAT,
        false, stride, normalOffset);

    _gl.bufferDataTyped(webgl.RenderingContext.ARRAY_BUFFER,
        new Float32List.fromList(buffer), webgl.RenderingContext.STATIC_DRAW);

    
    _gl.bindBuffer(webgl.RenderingContext.ELEMENT_ARRAY_BUFFER, _indexBuffer);

    _gl.bufferDataTyped(webgl.RenderingContext.ELEMENT_ARRAY_BUFFER, 
        new Int16List.fromList(indexData), webgl.RenderingContext.STATIC_DRAW);
   
  }

  int _viewportWidth;
  int _viewportHeight;
  static const TICS_PER_SECOND = 35;
  static const double FAR_DISTANCE = 200.0;

  void _render() {
    _gl.viewport(0, 0, _viewportWidth, _viewportHeight);
    _gl.clearColor(0, 0, 1, 1);
    //_gl.enable(webgl.RenderingContext.CULL_FACE);
    //_gl.cullFace(webgl.RenderingContext.BACK);
    _gl.clear(webgl.RenderingContext.COLOR_BUFFER_BIT |
        webgl.RenderingContext.DEPTH_BUFFER_BIT);

    _gl.bindBuffer(webgl.RenderingContext.ARRAY_BUFFER, _vertexBuffer);
    _gl.vertexAttribPointer(_aVertexPosition, 3, webgl.RenderingContext.FLOAT,
        false, stride, positionOffset);
    _gl.vertexAttribPointer(_aVertexNormal, 3, webgl.RenderingContext.FLOAT,
        false, stride, normalOffset);
    
    _gl.bindBuffer(webgl.RenderingContext.ELEMENT_ARRAY_BUFFER, _indexBuffer);
        

    // field of view is 90°, width-to-height ratio, hide things closer than 0.1 or further than 100
    _pMatrix = makePerspectiveMatrix(
        radians(90.0), _viewportWidth / _viewportHeight, 0.1, FAR_DISTANCE);
    //_pMatrix = new Matrix4.identity();

    _mvMatrix = new Matrix4.identity();
    _mvMatrix.translate(0.0, 0.0, -1.5);
    _mvMatrix.multiplyTranspose(_rotation);
    _mvMatrix.translate(-0.5, -0.5, -0.5);

    _setMatrixUniforms();

    //_gl.drawArrays(webgl.RenderingContext.TRIANGLES, 0, 6 * 6);
    //_gl.drawArrays(webgl.RenderingContext.TRIANGLES, 0, _vertexCount);
    
    //_gl.drawElements(webgl.RenderingContext.TRIANGLES, _vertexCount, webgl.RenderingContext.UNSIGNED_SHORT, 0);
    _gl.drawElements(webgl.RenderingContext.QUADS, _vertexCount, webgl.RenderingContext.UNSIGNED_SHORT, 0);
  }

  void _gameloop(Timer timer) {
    if (_needUpdate) {
      _render();
      _needUpdate = false;
    }
  }

  Timer startTimer() {
    _needUpdate = true;
    const duration = const Duration(milliseconds: 1000 ~/ TICS_PER_SECOND);

    return new Timer.periodic(duration, _gameloop);
  }
}

void main() {
  Renderer renderer = new Renderer(querySelector('#screen'));

  renderer.startTimer();
}
