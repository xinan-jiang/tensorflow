# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for diag_part and matrix_diag_part."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
import numpy

class DiagPartTest(xla_test.XLATestCase):

  def testPerformance(self):
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[1024, 1024])
        with self.test_scope():
          o = array_ops.diag_part(i)
        x = numpy.zeros([1024, 1024])
        params = {
            i: x,
        }
        result = o.eval(feed_dict=params)

        x = numpy.zeros([1024])
        self.assertAllEqual(x, result)

  def test2D(self):
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[3, 3])
        with self.test_scope():
          o = array_ops.diag_part(i)
        params = {
            i: [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([0, 4, 8], result)

  def test4D(self):
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2, 2, 2, 2])
        with self.test_scope():
          o = array_ops.diag_part(i)
        params = {
            i: [[[[0, 1], [2, 3]],
                 [[4, 5], [6, 7]]],
                [[[8, 9], [10, 11]],
                 [[12, 13], [14, 15]]]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[0, 5], [10, 15]], result)

class MatrixDiagPartTest(xla_test.XLATestCase):

  def testPerformance(self):
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[1024, 1024])
        with self.test_scope():
          o = array_ops.matrix_diag_part(i)
        x = numpy.zeros([1024, 1024])
        params = {
            i: x,
        }
        result = o.eval(feed_dict=params)

        x = numpy.zeros([1024])
        self.assertAllEqual(x, result)

  def test2D(self):
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[3, 3])
        with self.test_scope():
          o = array_ops.matrix_diag_part(i)
        params = {
            i: [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([0, 4, 8], result)

  def test3D(self):
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[3, 3, 3])
        with self.test_scope():
          o = array_ops.matrix_diag_part(i)
        params = {
            i: [[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                [[18, 19, 20], [21, 22, 23], [24, 25, 26]]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[0, 4, 8], [9, 13, 17], [18, 22, 26]], result)

  def test4D(self):
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[2, 2, 2, 2])
        with self.test_scope():
          o = array_ops.matrix_diag_part(i)
        params = {
            i: [[[[0, 1], [2, 3]],
                 [[4, 5], [6, 7]]],
                [[[8, 9], [10, 11]],
                 [[12, 13], [14, 15]]]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([[[0, 3], [4, 7]], [[8, 11], [12, 15]]], result)

  def test2DWithLargerMinorDimension(self):
    """Tests a matrix_diag_part where the minor dimension is larger then the
    major dimension.
    """
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[3, 4])
        with self.test_scope():
          o = array_ops.matrix_diag_part(i)
        params = {
            i: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([0, 5, 10], result)

  def test2DWithSmallerMinorDimension(self):
    """Tests a matrix_diag_part where the minor dimension is smaller then the
    major dimension.
    """
    for dtype in self.numeric_types:
      with self.session():
        i = array_ops.placeholder(dtype, shape=[4, 3])
        with self.test_scope():
          o = array_ops.matrix_diag_part(i)
        params = {
            i: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        }
        result = o.eval(feed_dict=params)

        self.assertAllEqual([0, 4, 8], result)

if __name__ == "__main__":
  googletest.main()
