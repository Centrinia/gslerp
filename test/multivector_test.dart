library multivector_test;

import 'package:vector_math/vector_math.dart';
import 'package:test/test.dart';
import '../web/geometric_algebra.dart';
import 'dart:math';

int bubbleSortSwaps(List<int> ap) {
  int count = 0;
  bool swapped;

  do {
    swapped = false;
    for (int i = 1; i < ap.length; i++) {
      if (ap[i - 1] > ap[i]) {
        int t = ap[i - 1];
        ap[i - 1] = ap[i];
        ap[i] = t;
        count++;
        swapped = true;
      }
    }
  } while (swapped);
  return count;
}

bool signFlipped(int a, int b) {
  List<int> ap = new List<int>();
  for (int i = 0; (a >> i) != 0; i++) {
    if ((a >> i) & 1 != 0) {
      ap.add(i << 1);
    }
  }
  for (int i = 0; (b >> i) != 0; i++) {
    if ((b >> i) & 1 != 0) {
      ap.add((i << 1) | 1);
    }
  }
  return bubbleSortSwaps(ap) & 1 != 0;
}
main() {
  test("canonical reordering sign", () {
    for (int a = 0; a < Multivector.DIMENSION; a++) {
      for (int b = 0; b < Multivector.DIMENSION; b++) {
        expect(signFlipped(a, b), equals(Multivector.reorderingSign(a, b)));
      }
    }
  });
  test("unit products", () {
    List<double> araw = new List<double>(Multivector.DIMENSION);
    for (int i = 0; i < Multivector.DIMENSION; i++) {
      araw[i] = (Multivector.DIMENSION + i).toDouble();
    }
    Multivector a = new Multivector.raw(araw);
    List<double> braw = new List<double>(Multivector.DIMENSION);

    for (int j = 0; j < Multivector.DIMENSION; j++) {
      for (int i = 0; i < Multivector.DIMENSION; i++) {
        braw[i] = 0.0;
      }
      braw[j] = 1.0;
      Multivector b = new Multivector.raw(braw);

      Multivector c = a * b;
      for (int i = 0; i < Multivector.DIMENSION; i++) {
        double t = c.at(i ^ j);
        t = signFlipped(i, j) ? -t : t;
        expect(t, equals(a.at(i)));
      }
    }
  });
  test("versor product", () {
    Multivector a = new Multivector(new Vector3(1.0, 0.0, 0.0));
    Multivector b = new Multivector(new Vector3(1.0, 1.0, 0.0));
    Multivector x = new Multivector(new Vector3(1.0, 2.0, 3.0));
    Multivector y = new Multivector(new Vector3(-4.0, 2.0, 6.0));
    expect((b * a) * x * (b * a).reverse(), equals(y));
  });
}
