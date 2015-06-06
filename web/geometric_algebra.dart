/* geometric_algebra.dart */

library geometric_algebra;

import 'dart:math';
import 'package:vector_math/vector_math.dart';

/**
 * Count the number of set bits in an int.
 */
int bitCount(int a) {
  int s = 0;
  int c = a;
  while (c != 0) {
    s += c & 1;
    c >>= 1;
  }
  return s;
}

class Multivector {
  static const VECTOR_DIMENSION = 3;
  static const DIMENSION = 1 << VECTOR_DIMENSION;
  List<double> _elements;
  
  /**
   * Construct the zero multivector.
   */
  Multivector.zero() {
    _elements = new List<double>(DIMENSION);
    for (int i = 0; i < DIMENSION; i++) {
      _elements[i] = 0.0;
    }
  }
  
  /** 
   * Construct a multivector from a list. The indexes are bit patterns of grades.
   */
  Multivector.raw(List<double> b) {
    _elements = new List<double>(DIMENSION);
    for (int i = 0; i < DIMENSION; i++) {
      _elements[i] = b[i];
    }
  }
  /**
   * Make a copy of a multivector.
   */
  Multivector.copy(Multivector b) {
    _elements = new List<double>(DIMENSION);
    for (int i = 0; i < DIMENSION; i++) {
      _elements[i] = b._elements[i];
    }
  }
  /**
   * Construct a random multivector with elements in [-1,1]
   */
  Multivector.random() {
    _elements = new List<double>(DIMENSION);
    Random r = new Random();

    for (int i = 0; i < DIMENSION; i++) {
      _elements[i] = r.nextDouble() * 2.0 - 1.0;
    }
  }
  /**
   * Extract a blade of specified grade from a multivector.
   */
  Multivector.blade(Multivector b, int grade) {
    _elements = new List<double>(DIMENSION);
    for (int i = 0; i < DIMENSION; i++) {
      if (bitCount(i) == grade) {
        _elements[i] = b._elements[i];
      } else {
        _elements[i] = 0.0;
      }
    }
  }
  /**
   * The multivector for the scalar value 1.
   */
  Multivector.one() {
    _elements = new List<double>(DIMENSION);
    _elements[0] = 1.0;
    for (int i = 1; i < DIMENSION; i++) {
      _elements[i] = 0.0;
    }
  }
  /**
   * Construct a coordinate basis vector of the designated axis.
   */
  Multivector.basisVector(int index) {
    _elements = new List<double>(DIMENSION);
    for (int i = 0; i < DIMENSION; i++) {
      _elements[i] = 0.0;
    }
    _elements[1 << index] = 1.0;
  }

  /**
   * Construct a multivector from a vector.
   */
  Multivector(Vector3 v) {
    _elements = new List<double>(DIMENSION);
    for (int i = 0; i < DIMENSION; i++) {
      _elements[i] = 0.0;
    }
    _elements[1 << 0] = v.x;
    _elements[1 << 1] = v.y;
    _elements[1 << 2] = v.z;
  }

  /**
   * Get a component.
   */
  double at(int i) {
    return _elements[i];
  }

  /**
   * The scalar component.
   */
  double get scalar {
    return _elements[0];
  }
  
  /**
   * The vector component.
   */
  Vector3 get vector {
    return new Vector3(_elements[1 << 0], _elements[1 << 1], _elements[1 << 2]);
  }

  /**
   * Determine the sign of the product of two multivectors.
   */
  static bool _reorderingSign(int a, int b) {
    a = a >> 1;
    int sum = 0;
    while (a != 0) {
      sum ^= (a & b);
      a = a >> 1;
    }
    int parity = 0;
    while (sum != 0) {
      parity ^= sum & 1;
      sum >>= 1;
    }
    return parity & 1 != 0;
  }

  /**
   * Compute the square of the norm of this multivector.
   */
  double get normSquare {
    Multivector c = this * this.reverse;
    return c.scalar;
  }

  /**
   * The norm of this multivector.
   */
  double get norm => sqrt(normSquare);

  /**
   * Get the reverse of this multivector.
   */
  Multivector get reverse {
    Multivector c = new Multivector.zero();
    for (int i = 0; i < DIMENSION; i++) {
      double t = _elements[i];
      c._elements[i] = (bitCount(i) & 3) >= 2 ? -t : t;
    }
    return c;
  }

  /**
   * Get the inverse of this multivector.
   */
  Multivector get inverse {
    double normsq = normSquare;

    return this.reverse.scale(1.0 / normsq);
  }
  /**
   * Add this multivector to another and return the result..
   */
  operator +(Multivector b) {
    Multivector c = new Multivector.zero();
    for (int i = 0; i < DIMENSION; i++) {
      c._elements[i] = _elements[i] + b._elements[i];
    }
    return c;
  }
  /**
   * Subtract another multivector from this and return the result.
   */
  operator -(Multivector b) {
    Multivector c = new Multivector.zero();
    
    for (int i = 0; i < DIMENSION; i++) {
      c._elements[i] = _elements[i] - b._elements[i];
    }
    return c;
  }

  /**
   * Multiply this multivector by a scalar.
   */
  Multivector scale(double b) {
    Multivector c = new Multivector.zero();

    for(int i=0;i<DIMENSION;i++){
      c._elements[i] = _elements[i] * b;
    }
    
    return c;
  }

  /**
   * Add a scalar.
   */
  Multivector addScalar(double b) {
    Multivector c = new Multivector.copy(this);
    c._elements[0] += b;
    return c;
  }

  /**
   * The geometric product.
   */
  operator *(Multivector b) {
    Multivector c = new Multivector.zero();

    for (int i = 0; i < DIMENSION; i++) {
      for (int j = 0; j < DIMENSION; j++) {
        double t = _elements[i] * b._elements[j];
        c._elements[i ^ j] += Multivector._reorderingSign(i, j) ? -t : t;
      }
    }
    return c;
  }

  /**
   * Divide this multivector.
   */
  operator /(Multivector b) {
    return this * b.inverse;
  }

  /**
   * Take e to the power of a blade to produce a versor.
   */
  Multivector versorExp() {
    double alphasq = min((this*this).scalar, 0.0);
    
    double alpha = sqrt(-alphasq);
    if (alpha == 0.0) {
      return new Multivector.one();
    } else {
      return this.scale(sin(alpha) / alpha).addScalar(cos(alpha));
    }
  }

  /**
   * Compute the logarithm of the versor.
   */
  Multivector versorLog() {
    Multivector c = new Multivector.blade(this, 2);
    double c2norm = c.norm;

    if (c2norm <= 0.0) {
      c = new Multivector.zero();
      if (this.scalar < 0) {
        /* We are asked to compute log(-1). Return a 360 degree rotation in an arbitrary plane. */

        c._elements[(1 << 0) + (1 << 1)] = PI;
      }
      return c;
    }
    double s = atan2(c2norm, _elements[0]);

    s = s - (s/PI).floor() * PI;
    s = [s-PI,s,s+PI].reduce((a,b) => a.abs() < b.abs() ? a : b);
    
    assert(s.abs() < (s+PI).abs() && s.abs() < (s-PI).abs());
       
    c = c.scale(s / c2norm);
    return c;
  }

  /**
   * Take this versor to a power.
   */
  Multivector versorPower(double b) {
    Multivector alog = this.versorLog();
    return alog.scale(b).versorExp();
  }

  String toString() {
    String out = "";
    for (int i = 0; i < DIMENSION; i++) {
      if (_elements[i] != 0) {
        String basis_vector = "";
        for (int j = 0; j < VECTOR_DIMENSION; j++) {
          if (((i >> j) & 1) != 0) {
            basis_vector += basis_vector != "" ? "^" : "";
            basis_vector += "e_" + (j + 1).toString();
          }
        }
        out += out != "" ? " + " : "";
        out += _elements[i].toString() + (i != 0 ? "*" : "") + basis_vector;
      }
    }
    out += out == "" ? "0" : "";
    return out;
  }
  
  /**
   * Construct a matrix from a versor.
   */
  Matrix4 fromVersor() {
    Multivector q = this;
    Multivector qinverse = q.inverse;
    Matrix3 rot = new Matrix3.columns(
        (q * new Multivector.basisVector(0) * qinverse).vector,
        (q * new Multivector.basisVector(1) * qinverse).vector,
        (q * new Multivector.basisVector(2) * qinverse).vector);

    Matrix4 out = new Matrix4.identity();
    out.setRotation(rot);
    
    return out;
  }

}
