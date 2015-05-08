// Copyright (c) 2015, <your name>. All rights reserved. Use of this source code
// is governed by a BSD-style license that can be found in the LICENSE file.

import 'dart:html';
import 'dart:web_gl' as webgl;
import 'dart:typed_data';
import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'geometric_algebra.dart';
import 'package:vector_math/vector_math.dart';


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

  static final int dimensions = 4;
  static final int stride = (dimensions * 4) * 2;
  static final int positionOffset = 0;
  static final int normalOffset = dimensions * 4;

  webgl.Buffer _vertexBuffer;
  webgl.Buffer _indexBuffer;

  int _vertexCount;
  Vector3 _center;
  double _scale;
  Point _startDrag;
  Multivector _rotation;
  double _rotationPower;
  static const ROTATION_POWER_START = 3.0;
  static const ROTATION_POWER = 1.002;
  List<Multivector> _keyframes;

  void resetKeyframes() {
    stopAnimation();
    _keyframes = new List<Multivector>();
    reset();
  }
  void appendKeyframe(int i) {
    stopAnimation();
    _keyframes.insert(i + 1, new Multivector.copy(_rotation));
  }
  void resetKeyframe(int i) {
    stopAnimation();
    _keyframes[i] = new Multivector.one();
    reset();
  }
  void saveKeyframe(int i) {
    stopAnimation();
    _keyframes[i] = new Multivector.copy(_rotation);
  }
  void loadKeyframe(int i) {
    stopAnimation();
    _rotation = new Multivector.copy(_keyframes[i]);
    update();
  }
  void removeKeyframe(int i) {
    stopAnimation();
    for (int j = i + 1; j < _keyframes.length; j++) {
      _keyframes[j - 1] = _keyframes[j];
    }
    _keyframes.removeLast();
  }

  Vector3 rotateVector(Vector3 a, Vector3 b, Vector3 x) {
    return x.reflect(a.normalized()).reflect((a + b).normalized());
  }
  Multivector makeRotation(Vector3 a, Vector3 b) {
    Vector3 bn = b.normalized();
    Vector3 an = a.normalized();
    Vector3 cn = (an + bn).normalized();

    Multivector aq = new Multivector(an);
    Multivector cq = new Multivector(cn);

    Multivector out = cq / aq;
    return out;
  }
  Multivector dragRotation(Point start, Point end,
      [double planeDistance = 1.0]) {
    Vector2 startDrag =
        new Vector2(start.x / _viewportWidth, 1.0 - start.y / _viewportHeight);
    Vector2 endDrag =
        new Vector2(end.x / _viewportWidth, 1.0 - end.y / _viewportHeight);

    startDrag -= new Vector2(0.5, 0.5);
    endDrag -= new Vector2(0.5, 0.5);
    Multivector rotation = makeRotation(
        new Vector3(startDrag.x, startDrag.y, planeDistance),
        new Vector3(endDrag.x, endDrag.y, planeDistance));

    return rotation;
  }
  void doDrag(Point next, bool endDrag, [double planeDistance = 1.0]) {
    Multivector rotation = dragRotation(_startDrag, next, planeDistance);

    _rotation = rotation.versorPower(_rotationPower) * _rotation;

    _rotationPower *= ROTATION_POWER;

    if (endDrag) {
      _startDrag = null;
    } else {
      _startDrag = next;
    }
    _needUpdate = true;
  }

  Renderer(CanvasElement canvas) {
    _keyframes = new List<Multivector>();

    _viewportWidth = canvas.width;
    _viewportHeight = canvas.height;

    _startDrag = null;
    _rotation = new Multivector.one();
    canvas.onMouseDown.listen((MouseEvent e) {
      _rotationPower = ROTATION_POWER_START;
      _startDrag = e.client;
      e.preventDefault();
    });

    canvas.onMouseMove.listen((MouseEvent e) {
      if (_startDrag != null) {
        doDrag(e.client, false);
        e.preventDefault();
      }
    });

    canvas.onMouseUp.listen((MouseEvent e) {
      if (_startDrag != null) {
        doDrag(e.client, true);
        e.preventDefault();
      }
    });
    canvas.onMouseOut.listen((MouseEvent e) {
      if (_startDrag != null) {
        doDrag(e.client, true);
        e.preventDefault();
      }
    });

    canvas.onTouchStart.listen((TouchEvent e) {
      _startDrag = e.touches.first.client;
      _rotationPower = ROTATION_POWER_START;

      e.preventDefault();
    });
    canvas.onTouchMove.listen((TouchEvent e) {
      if (_startDrag != null) {
        doDrag(e.touches.first.client, false);
        e.preventDefault();
      }
    });
    canvas.onTouchLeave.listen((TouchEvent e) {
      if (_startDrag != null) {
        doDrag(e.touches.first.client, true);
        e.preventDefault();
      }
    });
    canvas.onTouchCancel.listen((TouchEvent e) {
      if (_startDrag != null) {
        doDrag(e.touches.first.client, true);
        e.preventDefault();
      }
    });

    _gl = canvas.getContext("experimental-webgl");

    _initShaders();
    _needUpdate = false;

    HttpRequest.getString("models/bunny.json").then((String responseText) {
      Map data = JSON.decode(responseText);

      List<int> indexData = new List<int>();
      List<List<List<double>>> vertexData = new List<List<List<double>>>();
      for (var objectIndices in data["indices"]) {
        indexData.addAll(objectIndices);
      }
      for (int i = 0; i < data["vertices"].length; i++) {
        vertexData.add([data["vertices"][i], data["normals"][i]]);
      }
      _setupModel(vertexData, indexData);
      _needUpdate = true;
    });

    _gl.clearColor(0.0, 0.0, 0.0, 1.0);
    _gl.enable(webgl.RenderingContext.DEPTH_TEST);

    _gl.viewport(0, 0, _viewportWidth, _viewportHeight);
    _gl.clearColor(0, 0, 0, 1);
    _gl.enable(webgl.RenderingContext.CULL_FACE);
    _gl.cullFace(webgl.RenderingContext.BACK);
  }

  void _initShaders() {
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
    double det = nMatrix.determinant();
    nMatrix.invert();
    nMatrix *= pow(det, 1.0 / 3.0);
    nMatrix.copyIntoArray(tmpList);
    _gl.uniformMatrix3fv(_uNMatrix, false, tmpList);
  }

  void _setupModel(List<List<List<double>>> vertexes, List<int> indexData) {
    List<double> buffer = new List<double>();

    _vertexCount = indexData.length;

    _center = new Vector3(0.0, 0.0, 0.0);
    for (int i = 0; i < vertexes.length; i++) {
      Vector3 p =
          new Vector3.array(vertexes[i][0].map((x) => x.toDouble()).toList());

      buffer.addAll(p.storage);
      buffer.add(1.0);
      _center += p;

      Vector3 n =
          new Vector3.array(vertexes[i][1].map((x) => x.toDouble()).toList());
      buffer.addAll((-n).storage);

      buffer.add(0.0);
    }
    _center /= vertexes.length.toDouble();
    _scale = 0.0;
    for (int i = 0; i < vertexes.length; i++) {
      Vector3 p =
          new Vector3.array(vertexes[i][0].map((x) => x.toDouble()).toList());
      _scale += (p - _center).normalizeLength();
    }
    _scale /= vertexes.length.toDouble();
    _scale /= 2.0;

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
  static const TICS_PER_SECOND = 35.0;
  static const double FAR_DISTANCE = 200.0;

  Matrix4 fromVersor(Multivector q) {
    Matrix3 rot = new Matrix3.columns(
        (q * new Multivector.basisVector(0) * q.inverse()).vector,
        (q * new Multivector.basisVector(1) * q.inverse()).vector,
        (q * new Multivector.basisVector(2) * q.inverse()).vector);

    Matrix4 out = new Matrix4.identity();
    out.setRotation(rot);
    return out;
  }
  void _render() {
    _gl.clear(webgl.RenderingContext.COLOR_BUFFER_BIT |
        webgl.RenderingContext.DEPTH_BUFFER_BIT);

    // field of view is 90°, width-to-height ratio, hide things closer than 0.1 or further than 100
    _pMatrix = makePerspectiveMatrix(
        radians(90.0), _viewportWidth / _viewportHeight, 0.1, FAR_DISTANCE);

    Matrix4 model = new Matrix4.translation(-_center);
    Matrix4 rot = fromVersor(_rotation);

    model = rot * model;

    Matrix4 view = new Matrix4.identity();
    view.translate(0.0, 0.0, -1.5);
    Matrix4 s = new Matrix4.identity();
    s.scale(_scale);
    view = view * s;

    _mvMatrix = view * model;

    _setMatrixUniforms();

    _gl.drawElements(webgl.RenderingContext.TRIANGLES, _vertexCount,
        webgl.RenderingContext.UNSIGNED_SHORT, 0);
  }

  static int _binomialCoefficient(int n, int k) {
    int out = 1;
    for (int j = 1; j <= k; j++) {
      out *= n - k + j;
      out ~/= j;
    }
    return out;
  }

  double _animationLength;

  static Multivector _interpolateLinear(List<Multivector> keyframes, double t) {
    Multivector interpolateBetween(int interval, double x) {
      return keyframes[interval].versorLog().scale(1.0 - x) +
          keyframes[interval + 1].versorLog().scale(x);
    }
    int interval = (t * (keyframes.length - 1)).floor();

    double tk = t * (keyframes.length - 1) - interval;
    return interpolateBetween(interval, tk).versorExp();
  }
  static Multivector _interpolateHermite(
      List<Multivector> keyframes, double t) {
    // The tangent.
    Multivector interpolateBetween(int interval, double x) {
      double h00(double t) => (1 + 2.0 * t) * (1.0 - t) * (1.0 - t);
      double h10(double t) => t * (1.0 - t) * (1.0 - t);
      double h01(double t) => t * t * (3.0 - 2.0 * t);
      double h11(double t) => t * t * (t - 1.0);

      Multivector getTangent(int k) {
        double tension = 0.5;
        if (k == 0) {
          return (keyframes[k + 1].versorLog() - keyframes[k].versorLog())
              .scale(1.0 / (2.0 * ANIMATION_INTERVAL_LENGTH));
        } else if (k == keyframes.length - 1) {
          return (keyframes[k].versorLog() - keyframes[k - 1].versorLog())
              .scale(1.0 / (2.0 * ANIMATION_INTERVAL_LENGTH));
        } else {
          return (keyframes[k + 1].versorLog() - keyframes[k - 1].versorLog())
              .scale((1.0 - tension) / ANIMATION_INTERVAL_LENGTH);
        }
      }

      Multivector pk = keyframes[interval].versorLog();
      Multivector pk1 = keyframes[interval + 1].versorLog();
      Multivector mk = getTangent(interval);
      Multivector mk1 = getTangent(interval + 1);
      return pk.scale(h00(x)) +
          mk.scale(h10(x) * ANIMATION_INTERVAL_LENGTH) +
          pk1.scale(h01(x)) +
          mk1.scale(h11(x) * ANIMATION_INTERVAL_LENGTH);
    }

    int interval = (t * (keyframes.length - 1)).floor();

    double tk = t * (keyframes.length - 1) - interval;
    return interpolateBetween(interval, tk).versorExp();
  }

  static Multivector _interpolateBezier(List<Multivector> keyframes, double t) {
    double power = pow(1.0 - t, keyframes.length - 1.0);
    Multivector l = new Multivector.zero();
    for (int i = 0; i < keyframes.length; i++) {
      double coefficient =
          power * _binomialCoefficient(keyframes.length - 1, i);
      power *= t / (1.0 - t);
      l += keyframes[i].versorLog().scale(coefficient);
    }
    return l.versorExp();
  }

  String _method;
  set method(String val) => _method = val;

  void _gameloop(Timer timer) {
    if (_animationProgress != null) {
      while(cycleAnimation && _animationProgress>=_animationLength){
        _animationProgress -= _animationLength;
      }
      if (_animationProgress < _animationLength) {
        double t = _animationProgress / _animationLength;

        if (_method == 'linear') {
          _rotation = _interpolateLinear(_keyframes, t);
        } else if (_method == 'hermite') {
          _rotation = _interpolateHermite(_keyframes, t);
        } else if (_method == 'bezier') {
          _rotation = _interpolateBezier(_keyframes, t);
        }
        _render();
        _animationProgress += 1.0 / TICS_PER_SECOND;
      } else {
        _animationProgress = null;
        _needUpdate = false;
      }
    } else if (_needUpdate) {
      _render();
      _needUpdate = false;
    }
  }
  bool cycleAnimation = false;

  static const ANIMATION_INTERVAL_LENGTH = 2.0;
  double _animationProgress;
  void startAnimation() {
    if (_keyframes.length < 2) {
      return;
    }
    _animationLength = ANIMATION_INTERVAL_LENGTH * (_keyframes.length - 1);
    _animationProgress = 0.0;
  }
  void stopAnimation() {
    _animationProgress = null;
  }
  Timer startTimer() {
    const duration = const Duration(milliseconds: 1000 ~/ TICS_PER_SECOND);

    return new Timer.periodic(duration, _gameloop);
  }
  void update() {
    _needUpdate = true;
  }
  void reset() {
    _rotation = new Multivector.one();
    update();
  }
}

void main() {
  Renderer renderer = new Renderer(querySelector('#screen'));
  {
    SelectElement keyframeSelect =
        querySelector('#keyframeSelect') as SelectElement;

    querySelector('#addKeyframe').onClick.listen((MouseEvent e) {
      OptionElement option = new OptionElement();

      int index;
      if (keyframeSelect.options.length > 0) {
        index = keyframeSelect.selectedIndex + 1;
        if (index < keyframeSelect.options.length) {
          keyframeSelect.insertBefore(option, keyframeSelect.options[index]);
        } else {
          keyframeSelect.append(option);
        }
      } else {
        index = 0;
        keyframeSelect.append(option);
      }
      option.text = (index + 1).toString();
      option.value = index.toString();

      for (int i = index + 1; i < keyframeSelect.length; i++) {
        keyframeSelect.options[i].text = (i + 1).toString();
        keyframeSelect.options[i].value = i.toString();
      }

      renderer.appendKeyframe(index - 1);
      keyframeSelect.selectedIndex = index;
    });

    querySelector('#removeKeyframe').onClick.listen((MouseEvent e) {
      if (keyframeSelect.options.length > 0) {
        OptionElement option =
            keyframeSelect.options[keyframeSelect.selectedIndex];
        int index = int.parse(option.value);
        option.remove();
        for (int i = 0; i < keyframeSelect.options.length; i++) {
          OptionElement optioni = keyframeSelect.options[i];
          if (int.parse(optioni.value) > index) {
            optioni.value = (int.parse(optioni.value) - 1).toString();
            optioni.text = (int.parse(optioni.text) - 1).toString();
          }
        }
        if (index < keyframeSelect.length) {
          keyframeSelect.selectedIndex = index;
        } else {
          keyframeSelect.selectedIndex = keyframeSelect.length - 1;
        }
        renderer.removeKeyframe(index);
      }
    });

    querySelector('#saveKeyframe').onClick.listen((MouseEvent e) {
      OptionElement option =
          keyframeSelect.options[keyframeSelect.selectedIndex];
      int index = int.parse(option.value);

      renderer.saveKeyframe(index);
    });

    keyframeSelect.onChange.listen((Event e) {
      OptionElement option =
          keyframeSelect.options[keyframeSelect.selectedIndex];
      int index = int.parse(option.value);

      renderer.loadKeyframe(index);
    });

    querySelector('#revertKeyframe').onClick.listen((MouseEvent e) {
      if (keyframeSelect.options.length > 0) {
        OptionElement option =
            keyframeSelect.options[keyframeSelect.selectedIndex];
        int index = int.parse(option.value);

        renderer.resetKeyframe(index);
      } else {
        renderer.reset();
      }
    });

    querySelector('#resetKeyframes').onClick.listen((MouseEvent e) {
      renderer.resetKeyframes();
      for (OptionElement option in keyframeSelect.options) {
        option.remove();
      }
    });
  }
  {
CheckboxInputElement cycleElement = querySelector('#cycleOption');

cycleElement.onChange.listen((Event e){
 renderer.cycleAnimation = cycleElement.checked; 
});
renderer.cycleAnimation = cycleElement.checked; 
  }
  {
    SelectElement methodSelect =
        querySelector('#methodSelect') as SelectElement;
    methodSelect.onChange.listen((Event onData) {
      renderer.method = methodSelect.value;
    });
    methodSelect.value = 'hermite';
    renderer.method = methodSelect.value;
  }

  querySelector('#animateButton').onClick.listen((MouseEvent e) {
    renderer.startAnimation();
  });
  querySelector('#stopButton').onClick.listen((MouseEvent e) {
    renderer.stopAnimation();
  });

  renderer.startTimer();
}
