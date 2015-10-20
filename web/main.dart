// Copyright (c) 2015, <your name>. All rights reserved. Use of this source code
// is governed by a BSD-style license that can be found in the LICENSE file.

import 'dart:async';
import 'dart:convert';
import 'dart:html';
import 'dart:math';
import 'dart:typed_data';
import 'dart:web_gl' as webgl;

import 'package:vector_math/vector_math.dart';

import 'geometric_algebra.dart';

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

    cycleElement.onChange.listen((Event e) {
      renderer.cycleAnimation = cycleElement.checked;
    });
    renderer.cycleAnimation = cycleElement.checked;
  }
  {
    void selectHandler(
        String elementName, Function setValue, String defaultValue) {
      SelectElement element = querySelector(elementName) as SelectElement;
      element.onChange.listen((Event onData) {
        setValue(renderer, element.value);
      });
      element.value = defaultValue;
      setValue(renderer, element.value);
    }
    selectHandler(
        "#methodSelect", (r, v) => r.interpolationMethod = v, "hermite");
    selectHandler(
        "#modelSelect", (r, v) => r.modelFilename = v, "models/bunny.json");
  }

  {
    void changeValue(String elementName, Function valueChanger,
        Function getValue, double defaultValue) {
      InputElement valueElement = querySelector(elementName) as InputElement;
      valueElement.onChange.listen((Event onData) {
        try {
          valueChanger(renderer, double.parse(valueElement.value));
          renderer.update();
        } catch (e) {}
      });
      valueElement.onMouseWheel.listen((WheelEvent e) {
        const increment = 1.1;
        const scale = 200.0;
        double v = getValue(renderer);
        v *= pow(increment, -e.deltaY / scale);
        valueChanger(renderer, v);
        valueElement.value = v.toString();
        renderer.update();
      });

      valueElement.value = defaultValue.toString();
      valueChanger(renderer, defaultValue);
    }

    changeValue("#refractiveIndex", (r, v) => r.refractiveIndex = v,
        (r) => r.refractiveIndex, 1.2);
    changeValue("#transmittedLight", (r, v) => r.transmittedLight = v,
        (r) => r.transmittedLight, 0.0);
    changeValue("#reflectedLight", (r, v) => r.reflectedLight = v,
        (r) => r.reflectedLight, 1.0);
    changeValue("#diffuseLight", (r, v) => r.diffuseLight = v,
        (r) => r.diffuseLight, 0.0);
  }

  querySelector('#animateButton').onClick.listen((MouseEvent e) {
    renderer.startAnimation();
  });
  querySelector('#stopButton').onClick.listen((MouseEvent e) {
    renderer.stopAnimation();
  });

  renderer.startTimer();
}

/**
 * Compute the binomial coefficient n choose k.
 */
int binomialCoefficient(int n, int k) {
  int out = 1;
  for (int j = 1; j <= k; j++) {
    out *= n - k + j;
    out ~/= j;
  }
  return out;
}

class ShaderObject {
  webgl.Program _program;
  webgl.Shader _vertexShader;
  webgl.Shader _fragmentShader;

  webgl.RenderingContext _gl;
  int _count;
  int _stride;

  ShaderObject(webgl.RenderingContext gl) {
    _gl = gl;
  }

  void loadShaders(String _vertexShaderSource, String _fragmentShaderSource) {
    _vertexShader = _gl.createShader(webgl.RenderingContext.VERTEX_SHADER);
    _gl.shaderSource(_vertexShader, _vertexShaderSource);
    _gl.compileShader(_vertexShader);

    /**
     * Check if shaders were compiled properly. This is probably the most painful part
     * since there's no way to "debug" shader compilation
     */
    if (!_gl.getShaderParameter(
        _vertexShader, webgl.RenderingContext.COMPILE_STATUS)) {
      print(_gl.getShaderInfoLog(_vertexShader));
    }

    _fragmentShader = _gl.createShader(webgl.RenderingContext.FRAGMENT_SHADER);
    _gl.shaderSource(_fragmentShader, _fragmentShaderSource);
    _gl.compileShader(_fragmentShader);

    if (!_gl.getShaderParameter(
        _fragmentShader, webgl.RenderingContext.COMPILE_STATUS)) {
      print(_gl.getShaderInfoLog(_fragmentShader));
    }

    _program = _gl.createProgram();
    _gl.attachShader(_program, _vertexShader);
    _gl.attachShader(_program, _fragmentShader);
    _gl.linkProgram(_program);
    _gl.useProgram(_program);

    if (!_gl.getProgramParameter(
        _program, webgl.RenderingContext.LINK_STATUS)) {
      print(_gl.getProgramInfoLog(_program));
    }
  }

  Map<String, webgl.UniformLocation> get uniforms => _uniforms;

  webgl.Buffer _indexBuffer;
  webgl.Buffer _vertexBuffer;
  Map<String, Map> _attributes;
  Map<String, webgl.UniformLocation> _uniforms;

  void createUniforms(List<String> uniformNames) {
    _gl.useProgram(_program);

    _uniforms = new Map<String, webgl.UniformLocation>();
    for (String uniformName in uniformNames) {
      _uniforms[uniformName] = _gl.getUniformLocation(_program, uniformName);
    }
  }

  void createVertexBuffer(Map<String, Map> attributes) {
    _gl.useProgram(_program);

    _indexBuffer = _gl.createBuffer();
    _vertexBuffer = _gl.createBuffer();
    _attributes = new Map<String, Map>();
    for (String attributeName in attributes.keys) {
      int index = _gl.getAttribLocation(_program, attributeName);
      _gl.enableVertexAttribArray(index);

      Map attribute = new Map();
      attribute["index"] = index;
      attribute["size"] = attributes[attributeName]["size"];
      attribute["offset"] = attributes[attributeName]["offset"];
      _attributes[attributeName] = attribute;
    }
  }

  set stride(int stride) => _stride = stride;

  void bindBuffer() {
    const COMPONENT_BYTES = 4;
    _gl.useProgram(_program);

    /* The background model is just an oversized screen-filling triangle. */
    _gl.bindBuffer(webgl.RenderingContext.ARRAY_BUFFER, _vertexBuffer);
    for (String attributeName in _attributes.keys) {
      Map attribute = _attributes[attributeName];
      _gl.vertexAttribPointer(attribute["index"], attribute["size"],
          webgl.RenderingContext.FLOAT, false, _stride * COMPONENT_BYTES,
          attribute["offset"] * COMPONENT_BYTES);
    }

    _gl.bindBuffer(webgl.RenderingContext.ELEMENT_ARRAY_BUFFER, _indexBuffer);
  }

  void loadData(List<double> data, List<int> indexes) {
    _gl.useProgram(_program);

    _gl.bufferDataTyped(webgl.RenderingContext.ARRAY_BUFFER,
        new Float32List.fromList(data), webgl.RenderingContext.STATIC_DRAW);

    _gl.bufferDataTyped(webgl.RenderingContext.ELEMENT_ARRAY_BUFFER,
        new Int16List.fromList(indexes), webgl.RenderingContext.STATIC_DRAW);

    _count = indexes.length;
  }

  void render() {
    _gl.useProgram(_program);

    _gl.drawElements(webgl.RenderingContext.TRIANGLES, _count,
        webgl.RenderingContext.UNSIGNED_SHORT, 0);
  }

  Object withProgram(Function f) {
    _gl.useProgram(_program);
    return f();
  }
}

class Renderer {
  static const String BACKGROUND_VERTEX_SHADER = """
    attribute vec2 vPosition;
    uniform mat4 uIMVMatrix;
    uniform mat4 uPMatrix;
    varying vec3 fPosition;
    void main(void) {
      vec4 t =  uIMVMatrix * vec4(vPosition,-1.0,0.0);
      fPosition = t.xyz;
      gl_Position = vec4(vPosition,0.0,1.0);
    }
    """;
  static const String BACKGROUND_FRAGMENT_SHADER = """
    precision mediump float;
    varying vec3 fPosition;
    uniform samplerCube uSampler;
    void main(void) {
      vec4 fColor = textureCube(uSampler, fPosition);
      gl_FragColor = fColor;
    }
    """;

  static const String MODEL_VERTEX_SHADER = """
    attribute vec3 vPosition;
    attribute vec3 vNormal;
    uniform mat4 uMVMatrix;
    uniform mat3 uNMatrix;
    uniform mat4 uPMatrix;
    uniform mat4 uIMVMatrix;

    varying vec3 fPosition;
    varying vec3 fNormal;
    varying vec3 fColor;
    varying vec4 refracted;
    varying vec4 reflected;
    varying float lambert;

    uniform float uRefractiveIndex;

    void main(void) {
        fColor = vec3(1.0,1.0,1.0);
        
        vec3 normal_eye = normalize(uNMatrix * vNormal);
        vec3 position = vPosition;
        fNormal = normal_eye;

        vec3 position_eye = vec3(uMVMatrix * vec4(position,1.0));
        gl_Position = uPMatrix * vec4(position_eye,1.0);
        fPosition = position_eye;
        reflected = uIMVMatrix * vec4(reflect(position_eye,normal_eye),0.0);

        vec3 t = refract(position_eye.xyz, normal_eye, 1.0 / uRefractiveIndex);
        refracted = uIMVMatrix * vec4 (normalize(t), 0.0);

        lambert = dot(fNormal, -position_eye);
    }
    """;

  static const String MODEL_FRAGMENT_SHADER = """
    precision mediump float;
    varying vec3 fPosition;
    varying vec3 fNormal;
    varying vec3 fColor;
    varying vec4 refracted;
    varying vec4 reflected;
    varying float lambert;
    uniform samplerCube uSampler;
    uniform float uDiffuseLight;
    uniform float uReflectedLight;
    uniform float uTransmittedLight;

    void main(void) {
        float attenuation = 0.0;
        attenuation += max(0.0, lambert);
        vec3 color = attenuation * uDiffuseLight * fColor;
        // Reflected light.
        color += uReflectedLight * textureCube(uSampler, reflected.xyz).rgb;
        // Transmitted light.
        color += uTransmittedLight * textureCube(uSampler, refracted.xyz).rgb;
        gl_FragColor = vec4(color,1.0);
    }
    """;

  static const String IMAGE_DIR = "textures/";
  static const ANIMATION_INTERVAL_LENGTH = 2.0;

  static final CLEAR_COLOR = [0.0, 0.0, 0.0, 1.0];
  static const double TICS_PER_SECOND = 35.0;
  static const double NEAR_DISTANCE = 0.1;
  static const double FAR_DISTANCE = 2.0;

  static const int DIMENSIONS = 4;
  static const int STRIDE = (DIMENSIONS * 4) * 2;

  static const int POSITION_OFFSET = 0;
  static const int NORMAL_OFFSET = DIMENSIONS * 4;

  static const ROTATION_POWER_START = 1.0;
  static const ROTATION_POWER_FACTOR = 1.002;
  Matrix4 _perspectiveMatrix;
  Matrix4 _modelviewMatrix;

  webgl.Texture _texture;
  bool cycleAnimation = false;
  double _animationProgress;

  int _viewportWidth;
  int _viewportHeight;
  bool _ready;

  webgl.RenderingContext _gl;

  bool _needUpdate;

  /**
   * The world coordinate center of the model as weighed by the vertexes.
   */
  Vector3 _center;
  /**
   * The scaling factor for fitting the model on the screen. 
   */
  double _scale;
  /**
   * The starting point for a drag.
   */
  Vector2 _startDrag;
  Multivector _rotation;
  double _rotationPower;
  List<Multivector> _keyframes;

  double _animationLength;

  String _interpolationMethod;

  Renderer(CanvasElement canvas) {
    _ready = false;
    _keyframes = new List<Multivector>();

    _viewportWidth = canvas.width;
    _viewportHeight = canvas.height;

    _startDrag = null;
    _rotation = new Multivector.one();

    void startDragHandler(Event e, Point p) {
      _rotationPower = ROTATION_POWER_START;
      _startDrag = canvasToClip(p);
      e.preventDefault();
    }

    canvas.onMouseDown.listen((MouseEvent e) {
      startDragHandler(e, e.client);
    });

    canvas.onTouchStart.listen((TouchEvent e) {
      startDragHandler(e, e.touches.first.client);
    });

    void continueDragHandler(Event e, Point p, bool cancelling) {
      if (_startDrag != null) {
        doDrag(canvasToClip(p), cancelling);
        e.preventDefault();
      }
    }

    canvas.onMouseMove.listen((MouseEvent e) {
      continueDragHandler(e, e.client, false);
    });

    canvas.onTouchMove.listen((TouchEvent e) {
      continueDragHandler(e, e.touches.first.client, false);
    });

    canvas.onMouseUp.listen((MouseEvent e) {
      continueDragHandler(e, e.client, true);
    });
    canvas.onMouseOut.listen((MouseEvent e) {
      continueDragHandler(e, e.client, true);
    });

    canvas.onTouchLeave.listen((TouchEvent e) {
      continueDragHandler(e, e.touches.first.client, true);
    });
    canvas.onTouchCancel.listen((TouchEvent e) {
      continueDragHandler(e, e.touches.first.client, true);
    });

    _gl = canvas.getContext("webgl");

    _initShaders();
    _needUpdate = false;

    //_loadModel("models/bunny.json");

    _setupTextures();

    _gl.clearColor(
        CLEAR_COLOR[0], CLEAR_COLOR[1], CLEAR_COLOR[2], CLEAR_COLOR[3]);

    _gl.viewport(0, 0, _viewportWidth, _viewportHeight);
    _gl.clearColor(0, 0, 0, 1);
    _gl.enable(webgl.RenderingContext.CULL_FACE);
    _gl.cullFace(webgl.RenderingContext.BACK);
  }

  /**
   * The attenuation factor for the diffuse lighting component.
   */
  double _diffuseLight;
  get diffuseLight => _diffuseLight;
  set diffuseLight(double val) {
    _diffuseLight = val;

    _modelShader.withProgram(() {
      _gl.uniform1f(_modelShader.uniforms["uDiffuseLight"], _diffuseLight);
    });
  }
  set interpolationMethod(String val) => _interpolationMethod = val;

  /**
   * The name of the model file.
   */
  String _modelFilename;
  set modelFilename(String filename) {
    if (filename != _modelFilename) {
      _modelFilename = filename;
      _loadModel(filename);
    }
  }

  /**
   * The attenuation factor for the amount of light reflected from the model.
   */
  double _reflectedLight;
  get reflectedLight => _reflectedLight;
  set reflectedLight(double val) {
    _reflectedLight = val;
    _modelShader.withProgram(() {
      _gl.uniform1f(_modelShader.uniforms["uReflectedLight"], _reflectedLight);
    });
  }

  /**
   * The refractive index of the model.
   */
  double _refractiveIndex;
  get refractiveIndex => _refractiveIndex;
  set refractiveIndex(double val) {
    _refractiveIndex = val;
    _modelShader.withProgram(() {
      _gl.uniform1f(
          _modelShader.uniforms["uRefractiveIndex"], _refractiveIndex);
    });
  }
  /**
   * The attenuation factor for light that passes through the model.
   */
  double _transmittedLight;
  get transmittedLight => _transmittedLight;
  set transmittedLight(double val) {
    _transmittedLight = val;
    _modelShader.withProgram(() {
      _gl.uniform1f(
          _modelShader.uniforms["uTransmittedLight"], _transmittedLight);
    });
  }
  /**
   * Append the current keyframe to the list.
   */
  void appendKeyframe(int i) {
    stopAnimation();
    _keyframes.insert(i + 1, new Multivector.copy(_rotation));
  }
  /**
   * Convert HTML coordinates to clip coordinates. 
   */
  Vector2 canvasToClip(Point p) {
    return new Vector2(p.x / _viewportWidth, 1.0 - p.y / _viewportHeight) -
        new Vector2(0.5, 0.5);
  }
  /**
   * Perform the drag rotation. Set _rotation to the corresponding versor,
   * update the rotation speed, update the drag state, and request a rendering
   * update.
   */
  void doDrag(Vector2 next, bool endDrag, [double planeDistance = 1.0]) {
    Multivector rotation = dragRotation(_startDrag, next, planeDistance);

    _rotation = rotation.versorPower(_rotationPower) * _rotation;

    _rotationPower *= ROTATION_POWER_FACTOR;

    if (endDrag) {
      _startDrag = null;
    } else {
      _startDrag = next;
    }
    _needUpdate = true;
  }
  /**
   * Construct a versor from two 2D clip space vectors and a z-coordinate for their common plane.
   */
  Multivector dragRotation(Vector2 start, Vector2 end,
      [double planeDistance = 1.0]) {
    return makeRotation(new Vector3(start.x, start.y, planeDistance),
        new Vector3(end.x, end.y, planeDistance));
  }
  void loadKeyframe(int i) {
    stopAnimation();
    _rotation = new Multivector.copy(_keyframes[i]);
    update();
  }

  /**
   * Remove the currently selected keyframe.
   */
  void removeKeyframe(int i) {
    stopAnimation();
    for (int j = i + 1; j < _keyframes.length; j++) {
      _keyframes[j - 1] = _keyframes[j];
    }
    _keyframes.removeLast();
  }

  /**
   * Reset the current model transform to the identity.
   */
  void reset() {
    _rotation = new Multivector.one();
    update();
  }

  /**
   * Set the current keyframe to the identity and reset the current model transform.
   */
  void resetKeyframe(int i) {
    stopAnimation();
    _keyframes[i] = new Multivector.one();
    reset();
  }

  /**
   * Remove all keyframes and reset the current model transform.
   */
  void resetKeyframes() {
    stopAnimation();
    _keyframes = new List<Multivector>();
    reset();
  }

  /**
   * Set the currently selected keyframe to the current model transform. 
   */
  void saveKeyframe(int i) {
    stopAnimation();
    _keyframes[i] = new Multivector.copy(_rotation);
  }

  /**
   * Start the interpolation animation.
   */
  void startAnimation() {
    if (_keyframes.length < 2) {
      return;
    }
    _animationLength = ANIMATION_INTERVAL_LENGTH * (_keyframes.length - 1);
    _animationProgress = 0.0;
  }

  /**
   * Start the rendering loop.
   */
  Timer startTimer() {
    const duration = const Duration(milliseconds: 1000 ~/ TICS_PER_SECOND);

    return new Timer.periodic(duration, _gameloop);
  }

  /**
   * Stop the current animation if it is occurring.
   */
  void stopAnimation() {
    _animationProgress = null;
  }

  /**
   * Demand a rendering update.
   */
  void update() {
    _needUpdate = true;
  }
  /** 
   * Produce one frame.
   */
  void _gameloop(Timer timer) {
    if (!_ready) {
      return;
    }
    if (_animationProgress != null) {
      while (cycleAnimation && _animationProgress >= _animationLength) {
        _animationProgress -= _animationLength;
      }
      if (_animationProgress < _animationLength) {
        double t = _animationProgress / _animationLength;

        if (_interpolationMethod == 'linear') {
          _rotation = _interpolateLinear(_keyframes, t);
        } else if (_interpolationMethod == 'hermite') {
          _rotation = _interpolateHermite(_keyframes, t);
        } else if (_interpolationMethod == 'bezier') {
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

  ShaderObject _backgroundShader;
  ShaderObject _modelShader;

  /**
   * Set and initialize the shaders.
   */
  void _initShaders() {
    _backgroundShader = new ShaderObject(_gl);

    _backgroundShader.loadShaders(
        BACKGROUND_VERTEX_SHADER, BACKGROUND_FRAGMENT_SHADER);

    _backgroundShader.stride = 2;
    _backgroundShader
        .createVertexBuffer({"vPosition": {"size": 2, "offset": 0}});

    _backgroundShader.bindBuffer();

    _backgroundShader.createUniforms(["uPMatrix", "uIMVMatrix", "uSampler"]);

    _modelShader = new ShaderObject(_gl);
    _modelShader.loadShaders(MODEL_VERTEX_SHADER, MODEL_FRAGMENT_SHADER);
    _modelShader.stride = 4 * 2;

    _modelShader.createVertexBuffer({
      //
      "vPosition": {"size": 4, "offset": 0},
      //
      "vNormal": {"size": 3, "offset": 4}
    });

    _modelShader.createUniforms([
      "uPMatrix",
      "uMVMatrix",
      "uIMVMatrix",
      "uNMatrix",
      "uDiffuseLight",
      "uReflectedLight",
      "uTransmittedLight",
      "uRefractiveIndex"
    ]);
  }

  void _loadModel(String filename) {
    HttpRequest.getString(filename).then((String responseText) {
      Map data = JSON.decode(responseText);

      List<int> indexData = new List<int>();
      List<List<List<double>>> vertexData = new List<List<List<double>>>();
      for (var objectIndices in data["indices"]) {
        indexData.addAll(objectIndices);
      }
      for (int i = 0; i < data["vertices"].length; i++) {
        vertexData
            .add([data["vertices"][i], data["normals"][i].map((x) => -x)]);
      }
      _setupModel(vertexData, indexData);

      _needUpdate = true;
    });
  }

  /**
   * Render the scene.
   */
  void _render() {
    _gl.clear(webgl.RenderingContext.COLOR_BUFFER_BIT |
        webgl.RenderingContext.DEPTH_BUFFER_BIT);

    /* field of view is 90°, width-to-height ratio, hide things closer than 0.1 or further than 100 */
    _perspectiveMatrix = makePerspectiveMatrix(radians(90.0),
        _viewportWidth / _viewportHeight, NEAR_DISTANCE, FAR_DISTANCE);

    Matrix4 model = new Matrix4.translation(-_center);
    Matrix4 rot = _rotation.fromVersor();

    model = rot * model;

    Matrix4 view = new Matrix4.identity();
    view.translate(0.0, 0.0, -1.5);
    Matrix4 s = new Matrix4.identity();
    s.scale(_scale);
    view = view * s;

    _modelviewMatrix = view * model;

    _setMatrixUniforms();

    /* Render the background. */
    _gl.disable(webgl.RenderingContext.DEPTH_TEST);

    _gl.activeTexture(webgl.TEXTURE0);

    _backgroundShader.bindBuffer();
    _backgroundShader.render();

    /* Render the model. */
    _gl.enable(webgl.RenderingContext.DEPTH_TEST);
    _modelShader.bindBuffer();
    _modelShader.render();
  }

  /**
   * Update the modelview and perspective matrices.
   */
  void _setMatrixUniforms() {
    Float32List perspectiveList = new Float32List(16);
    _perspectiveMatrix.copyIntoArray(perspectiveList);

    Float32List modelviewList = new Float32List(16);
    _modelviewMatrix.copyIntoArray(modelviewList);
    
    Float32List inverseModelviewList = new Float32List(16);

    Matrix4 t = new Matrix4.copy(_modelviewMatrix);
    t.invert();
    t.copyIntoArray(inverseModelviewList);

    Float32List normalList = new Float32List(9);
    Matrix3 nMatrix = new Matrix3.columns(_modelviewMatrix.row0.xyz,
        _modelviewMatrix.row1.xyz, _modelviewMatrix.row2.xyz);
    double det = nMatrix.determinant();
    nMatrix.invert();
    nMatrix *= pow(det, 1.0 / 3.0);
    nMatrix.copyIntoArray(normalList);

    _modelShader.withProgram(() {
      _gl.uniformMatrix4fv(
          _modelShader.uniforms["uPMatrix"], false, perspectiveList);
      _gl.uniformMatrix4fv(
          _modelShader.uniforms["uMVMatrix"], false, modelviewList);
      _gl.uniformMatrix4fv(
          _modelShader.uniforms["uIMVMatrix"], false, inverseModelviewList);
      _gl.uniformMatrix3fv(
          _modelShader.uniforms["uNMatrix"], false, normalList);
    });

    _backgroundShader.withProgram(() {
      _gl.uniformMatrix4fv(
          _backgroundShader.uniforms["uPMatrix"], false, perspectiveList);
      _gl.uniformMatrix4fv(_backgroundShader.uniforms["uIMVMatrix"], false,
          inverseModelviewList);
    });
  }
  void _setupModel(List<List<List<double>>> vertexes, List<int> indexData) {
    List<double> buffer = new List<double>();

    /* Compute the center of the model. */
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

    /* The scale of the model is the half the average distance to the center. */
    _scale = 0.0;
    for (int i = 0; i < vertexes.length; i++) {
      Vector3 p =
          new Vector3.array(vertexes[i][0].map((x) => x.toDouble()).toList());
      _scale += (p - _center).normalizeLength();
    }
    _scale /= vertexes.length.toDouble() * 2.0;

    _modelShader.bindBuffer();
    _modelShader.loadData(buffer, indexData);

    /* The background model is just an oversized screen-filling triangle. */

    _backgroundShader.bindBuffer();
    _backgroundShader.loadData([
      // Lower left.
      -1.0,
      -1.0,
      // Lower right.
      3.0,
      -1.0,
      // Upper Left.
      -1.0,
      3.0,

      //
      1.0,
      -1.0,
      -1.0,
      1.0,
      1.0,
      1.0
    ], [0, 3, 4, 4, 3, 5]);
  }
  /**
   * Load the cubemap.
   */
  void _setupTextures() {
    List<String> imageDescriptions = [
      {
        'filename': 'negx.jpg',
        'side': webgl.RenderingContext.TEXTURE_CUBE_MAP_NEGATIVE_X
      },
      {
        'filename': 'posx.jpg',
        'side': webgl.RenderingContext.TEXTURE_CUBE_MAP_POSITIVE_X
      },
      {
        'filename': 'negy.jpg',
        'side': webgl.RenderingContext.TEXTURE_CUBE_MAP_NEGATIVE_Y
      },
      {
        'filename': 'posy.jpg',
        'side': webgl.RenderingContext.TEXTURE_CUBE_MAP_POSITIVE_Y
      },
      {
        'filename': 'negz.jpg',
        'side': webgl.RenderingContext.TEXTURE_CUBE_MAP_NEGATIVE_Z
      },
      {
        'filename': 'posz.jpg',
        'side': webgl.RenderingContext.TEXTURE_CUBE_MAP_POSITIVE_Z
      }
    ];

    _gl.activeTexture(webgl.TEXTURE0);

    _modelShader.withProgram(() {
      _texture = _gl.createTexture();
      _gl.bindTexture(webgl.TEXTURE_CUBE_MAP, _texture);
      _gl.uniform1i(_modelShader.uniforms["uSampler"], 0);
      _gl.texParameteri(
          webgl.TEXTURE_CUBE_MAP, webgl.TEXTURE_MAG_FILTER, webgl.LINEAR);
      _gl.texParameteri(
          webgl.TEXTURE_CUBE_MAP, webgl.TEXTURE_MIN_FILTER, webgl.LINEAR);
    });

    _backgroundShader.withProgram(() {
      _gl.bindTexture(webgl.TEXTURE_CUBE_MAP, _texture);
      _gl.uniform1i(_backgroundShader.uniforms["uSampler"], 0);
      _gl.texParameteri(
          webgl.TEXTURE_CUBE_MAP, webgl.TEXTURE_MAG_FILTER, webgl.LINEAR);
      _gl.texParameteri(
          webgl.TEXTURE_CUBE_MAP, webgl.TEXTURE_MIN_FILTER, webgl.LINEAR);
    });

    int ready = 0;
    for (Map imageDescription in imageDescriptions) {
      String imageName = imageDescription['filename'];
      int side = imageDescription['side'];
      ImageElement image = new ImageElement();
      //bool imageReady = false;
      image.onLoad.listen((e) {
        _gl.texImage2DImage(
            side, 0, webgl.RGBA, webgl.RGBA, webgl.UNSIGNED_BYTE, image);
        ready++;
        if (ready == 6) {
          _ready = true;
        }
      }, onError: (e) => print(e));
      image.src = IMAGE_DIR + imageName;
    }
  }
  /**
   * Produce a versor that rotates vector a to vector b. 
   */
  static Multivector makeRotation(Vector3 a, Vector3 b) {
    Vector3 bn = b.normalized();
    Vector3 an = a.normalized();
    Vector3 cn = (an + bn).normalized();

    Multivector aq = new Multivector(an);
    Multivector cq = new Multivector(cn);

    Multivector out = cq / aq;
    return out;
  }
  static Multivector _interpolateBezier(List<Multivector> keyframes, double t) {
    double power = pow(1.0 - t, keyframes.length - 1.0);
    Multivector l = new Multivector.zero();
    for (int i = 0; i < keyframes.length; i++) {
      double coefficient = power * binomialCoefficient(keyframes.length - 1, i);
      power *= t / (1.0 - t);
      l += keyframes[i].versorLog().scale(coefficient);
    }
    return l.versorExp();
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
  /**
   * Perform a linear interpolation between two frames.
   */
  static Multivector _interpolateLinear(List<Multivector> keyframes, double t) {
    Multivector interpolateBetween(int interval, double x) {
      return keyframes[interval].versorLog().scale(1.0 - x) +
          keyframes[interval + 1].versorLog().scale(x);
    }
    int interval = (t * (keyframes.length - 1)).floor();

    double tk = t * (keyframes.length - 1) - interval;
    return interpolateBetween(interval, tk).versorExp();
  }
}
