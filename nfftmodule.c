#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>
#include <nfft3.h>
#include <math.h>
#include <stdio.h>
#include <complex.h>

//#include "nfftclassmodule.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

PyDoc_STRVAR(ndfft__doc__, "ndfft(values, coordinates, output_shape, pixel_size)\n\nCalculate the fft of arbitrarily spaced points.\n\values should be an  array of length N.\ncoordinates shoulb be an NxD dimensional array.\noutput_shape is a tuple indicating the dimensions of the output array\npixel_size is a float describing the pixel spacing of the output array");
static PyObject *ndfft(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *in_obj, *coord_obj;
  PyObject *use_direct_obj = NULL;
  PyObject *shape_tuple_obj;
  //PyTupleObject *shape_tuple;
  double pixel_size;

  static char *kwlist[] = {"values", "coordinates", "output_shape", "pixel_size", "use_direct", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOd|O", kwlist, &in_obj, &coord_obj, &shape_tuple_obj,
				   &pixel_size, &use_direct_obj)) {
    return NULL;
  }
  int use_direct = 0;
  if (use_direct_obj != NULL && PyObject_IsTrue(use_direct_obj)) {
    use_direct = 1;
  }
  
  int number_of_dimensions = PyTuple_Size(shape_tuple_obj);
  npy_intp *shape = malloc(number_of_dimensions*sizeof(npy_intp));
  for (int i = 0; i < number_of_dimensions; i++) {
    shape[i] = (npy_intp) PyInt_AsLong(PyTuple_GetItem(shape_tuple_obj, i));
  }
    
  PyObject *coord_array = PyArray_FROM_OTF(coord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *in_array = PyArray_FROM_OTF(in_obj, NPY_COMPLEX128, NPY_IN_ARRAY);

  if (coord_array == NULL || in_array == NULL) {
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  int ndim = PyArray_NDIM(in_array);
  if (ndim != 1) {
    PyErr_SetString(PyExc_ValueError, "Input array must be 1 dimensional\n");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  if ((PyArray_NDIM(coord_array) != 2 ||
       PyArray_DIM(coord_array, 1) != ndim ||
       PyArray_DIM(coord_array, 0) != PyArray_DIM(in_array, 0)) &&
      (PyArray_NDIM(coord_array) != 1 || number_of_dimensions != 1 || PyArray_DIM(coord_array, 0) != PyArray_DIM(in_array, 0))) {
    PyErr_SetString(PyExc_ValueError, "Values and coordinates must have the same length");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  if (number_of_dimensions != 1 && PyArray_DIM(coord_array, 1) != PyArray_DIM(in_array, 0)) {
    PyErr_SetString(PyExc_ValueError, "Dimensionality of coordinates and shape must match.");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  PyObject *out_array = (PyObject *)PyArray_SimpleNew(number_of_dimensions, shape, NPY_COMPLEX128);
  
  int number_of_points = (int) PyArray_DIM(coord_array, 0);

  nfft_plan my_plan;

  int *shape_int = malloc(number_of_dimensions*sizeof(int));
  for (int i = 0; i < number_of_dimensions; i++) {
    shape_int[i] = shape[i];
  }
  nfft_init(&my_plan, number_of_dimensions, shape_int, number_of_points);
  memcpy(my_plan.f, PyArray_DATA(in_array), number_of_points*sizeof(fftw_complex));
  //memcpy(my_plan.x, PyArray_DATA(coord_array), number_of_dimensions*number_of_points*sizeof(double));
  double *coord_array_data = (double *) PyArray_DATA(coord_array);
  for (int i = 0; i < number_of_dimensions*number_of_points; i++) {
    my_plan.x[i] = coord_array_data[i] * pixel_size;
  }

  if (my_plan.flags &PRE_ONE_PSI) {
    nfft_precompute_one_psi(&my_plan);
  }
  
  if (my_plan.flags &PRE_PSI) {
    nfft_precompute_psi(&my_plan);
  }

  if (my_plan.flags &PRE_FULL_PSI) {
    nfft_precompute_full_psi(&my_plan);
  }

  if (my_plan.flags &PRE_LIN_PSI) {
    nfft_precompute_lin_psi(&my_plan);
  }
  
  if (use_direct == 1) {
    nfft_adjoint_direct(&my_plan);
  } else {
    nfft_adjoint(&my_plan);
  }

  //memcpy(PyArray_DATA(out_array), my_plan.f_hat, ((int)PyArray_SIZE(out_array))*sizeof(fftw_complex));
  int output_size = ((int)PyArray_SIZE(out_array));
  //fftw_complex *out_array_data = (fftw_complex *) PyArray_DATA(out_array);

  /* We are doing an inverse transform to get the nonequispaced to equispaced. Therefore
     we have to conjugate the output since we are actually wanting a forward transform. */
  double *out_array_data = (double *) PyArray_DATA(out_array);
  double *f_hat_as_double = (double *) my_plan.f_hat;
  for (int i = 0; i < output_size; i++) {
    out_array_data[2*i] = f_hat_as_double[2*i];
    out_array_data[2*i+1] = -f_hat_as_double[2*i+1];
    //printf("%g, %g\n", out_array_data[2*i], out_array_data[2*i+1]);
  }

  nfft_finalize(&my_plan);
  free(shape);

  Py_XDECREF(coord_array);
  Py_XDECREF(in_array);
  
  return out_array;
}

PyDoc_STRVAR(nfft__doc__, "nfft(real_space, pixel_size, coordinates)\n\nCalculate nfft from arbitrary dimensional array.\n\real_space should be an array (or any object that can trivially be converted to one.\nreal_space is a float indicating the pixel size in the input arrray\ncoordinates should be a NxD array where N is the number of points where the Fourier transform should be evaluated and D is the dimensionality of the input array\ndirect (optional) requires the use of the more accurate but slower ndft (default is False)");
static PyObject *nfft(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *in_obj, *coord_obj;
  PyObject *use_direct_obj = NULL;
  double pixel_size;
  
  static char *kwlist[] = {"real_space", "pixel_size", "coordinates", "use_direct", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OdO|O", kwlist, &in_obj, &pixel_size, &coord_obj, &use_direct_obj)) {
    return NULL;
  }
  int use_direct = 0;
  if (use_direct_obj != NULL && PyObject_IsTrue(use_direct_obj)) {
    use_direct = 1;
  }

  PyObject *coord_array = PyArray_FROM_OTF(coord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *in_array = PyArray_FROM_OTF(in_obj, NPY_COMPLEX128, NPY_IN_ARRAY);
  if (coord_array == NULL || in_array == NULL) {
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  int ndim = PyArray_NDIM(in_array);
  if (ndim <= 0) {
    PyErr_SetString(PyExc_ValueError, "Input array can't be 0 dimensional\n");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  if ((PyArray_NDIM(coord_array) != 2 || PyArray_DIM(coord_array, 1) != ndim) && (ndim != 1 || PyArray_NDIM(coord_array) != 1)) {
    PyErr_SetString(PyExc_ValueError, "Coordinates must be given as array of dimensions [NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS] of [NUMBER_OF_POINTS for 1D transforms.\n");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }
  int number_of_points = (int) PyArray_DIM(coord_array, 0);

  nfft_plan my_plan;
  int total_number_of_pixels = 1;
  int dims[ndim];
  int dim;
  for (dim = 0; dim < ndim; ++dim) {
    dims[dim] = (int)PyArray_DIM(in_array, dim);
    total_number_of_pixels *= dims[dim];
  }
  nfft_init(&my_plan, ndim, dims, number_of_points);
  memcpy(my_plan.f_hat, PyArray_DATA(in_array), total_number_of_pixels*sizeof(fftw_complex));
  //memcpy(my_plan.x, PyArray_DATA(coord_array), ndim*number_of_points*sizeof(double));
  double *coord_array_data = (double *) PyArray_DATA(coord_array);
  for (int i = 0; i < ndim*number_of_points; i++) {
    my_plan.x[i] = coord_array_data[i] * pixel_size;
  }
  
  if (my_plan.flags &PRE_ONE_PSI) {
    nfft_precompute_one_psi(&my_plan);
  }
  
  if (my_plan.flags &PRE_PSI) {
    nfft_precompute_psi(&my_plan);
  }

  if (my_plan.flags &PRE_FULL_PSI) {
    nfft_precompute_full_psi(&my_plan);
  }

  if (my_plan.flags &PRE_LIN_PSI) {
    nfft_precompute_lin_psi(&my_plan);
  }

  if (use_direct == 1) {
    nfft_trafo_direct(&my_plan);
  } else {
    nfft_trafo(&my_plan);
  }

  //int out_dim[] = {number_of_points};
  npy_intp *out_dim = malloc(1*sizeof(npy_intp));
  out_dim[0] = number_of_points;
  //PyObject *out_array = (PyObject *)PyArray_FromDims(1, out_dim, NPY_COMPLEX128);
  PyObject *out_array = (PyObject *)PyArray_SimpleNew(1, out_dim, NPY_COMPLEX128);
  memcpy(PyArray_DATA(out_array), my_plan.f, number_of_points*sizeof(fftw_complex));

  Py_XDECREF(coord_array);
  Py_XDECREF(in_array);

  nfft_finalize(&my_plan);
  return out_array;
}

PyDoc_STRVAR(nfft_inplace__doc__, "nfft_inplace(real_space, coordinates)\n\nCalculate nfft from arbitrary dimensional array.\n\real_space should be an array (or any object that can trivially be converted to one.\nreal_space is a float indicating the pixel_size of the input array\ncoordinates should be a NxD array where N is the number of points where the Fourier transform should be evaluated and D is the dimensionality of the input array\noutput_array should be ndarray of type complex128. The is written to here, if the array is a continuous block in memory this can speed up the calculation.\ndirect (optional) requires the use of the more accurate but slower ndft (default is False).");
static PyObject *nfft_inplace(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *in_obj, *coord_obj, *out_obj;
  PyObject *use_direct_obj = NULL;
  double pixel_size;

  static char *kwlist[] = {"real_space", "pixel_size", "coordinates", "output", "use_direct", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OdOO|O", kwlist, &in_obj, &pixel_size, &coord_obj, &out_obj, &use_direct_obj)) {
    return NULL;
  }
  int use_direct = 0;
  if (use_direct_obj != NULL && PyObject_IsTrue(use_direct_obj)) {
    use_direct = 1;
  }

  PyObject *coord_array = PyArray_FROM_OTF(coord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *in_array = PyArray_FROM_OTF(in_obj, NPY_COMPLEX128, NPY_IN_ARRAY);
  if (coord_array == NULL || in_array == NULL) {
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  int ndim = PyArray_NDIM(in_array);
  if (ndim <= 0) {
    PyErr_SetString(PyExc_ValueError, "Input array can't be 0 dimensional\n");
    return NULL;
  }

  if ((PyArray_NDIM(coord_array) != 2 || PyArray_DIM(coord_array, 1) != ndim) && (ndim != 1 || PyArray_NDIM(coord_array) != 1)) {
    PyErr_SetString(PyExc_ValueError, "Coordinates must be given as array of dimensions [NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS] of [NUMBER_OF_POINTS for 1D transforms.\n");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }
  int number_of_points = (int) PyArray_DIM(coord_array, 0);

  if (!PyArray_Check(out_obj)) {
    PyErr_SetString(PyExc_ValueError, "Output must be numpy.array of dtype complex128");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  if (PyArray_DESCR(out_obj)->type_num != NPY_COMPLEX128) {
    PyErr_SetString(PyExc_ValueError, "Output array must be of dtype complex128");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    return NULL;
  }

  PyObject *out_array = PyArray_FROM_OTF(out_obj, NPY_COMPLEX128, NPY_INOUT_ARRAY);
  if (PyArray_NDIM(out_array) != 1 || PyArray_DIM(out_array, 0) != number_of_points) {
    PyErr_SetString(PyExc_ValueError,  "Output array must be one dimensional and same length as coordinates array.");
    Py_XDECREF(coord_array);
    Py_XDECREF(in_array);
    Py_XDECREF(out_array);
    return NULL;
  }

  nfft_plan my_plan;
  int total_number_of_pixels = 1;
  int dims[ndim];
  int dim;
  for (dim = 0; dim < ndim; ++dim) {
    dims[dim] = (int)PyArray_DIM(in_array, dim);
    total_number_of_pixels *= dims[dim];
  }
  nfft_init(&my_plan, ndim, dims, number_of_points);
  memcpy(my_plan.f_hat, PyArray_DATA(in_array), total_number_of_pixels*sizeof(fftw_complex));
  //memcpy(my_plan.x, PyArray_DATA(coord_array), ndim*number_of_points*sizeof(double));
  double *coord_array_data = (double *) PyArray_DATA(coord_array);
  for (int i = 0; i < ndim*number_of_points; i++) {
    my_plan.x[i] = coord_array_data[i] * pixel_size;
  }

  if (my_plan.flags &PRE_ONE_PSI) {
    nfft_precompute_one_psi(&my_plan);
  }
  
  if (my_plan.flags &PRE_PSI) {
    nfft_precompute_psi(&my_plan);
  }

  if (my_plan.flags &PRE_FULL_PSI) {
    nfft_precompute_full_psi(&my_plan);
  }

  if (my_plan.flags &PRE_LIN_PSI) {
    nfft_precompute_lin_psi(&my_plan);
  }

  if (use_direct == 1) {
    nfft_trafo_direct(&my_plan);
  } else {
    nfft_trafo(&my_plan);
  }
  memcpy(PyArray_DATA(out_array), my_plan.f, number_of_points*sizeof(fftw_complex));

  Py_XDECREF(coord_array);
  Py_XDECREF(in_array);

  nfft_finalize(&my_plan);
  return Py_BuildValue("i", 1);
}

PyDoc_STRVAR(Transformer__doc__, "Transformer(real_space)\n\nCreates an object that can be used to calculate multiple nfft transforms from the same array.");
typedef struct {
  PyObject_HEAD
  nfft_plan my_plan;
  PyArrayObject *real_map;
  double pixel_size;
  fftw_complex *sneaky_ref;
  int ndim;
  int max_number_of_points;
}Transformer;

static PyMemberDef Transformer_members[] = {
  {"real_map", T_OBJECT_EX, offsetof(Transformer, real_map), 0, "Real space map."},
  {"pixel_size", T_DOUBLE, offsetof(Transformer, pixel_size), 0, "Pixel size"},
  {NULL}
};

static int Transformer_init(Transformer *self, PyObject *args, PyObject *kwds)
{
  PyObject *input_obj;
  if (!PyArg_ParseTuple(args, "Odi", &input_obj, &self->pixel_size, &self->max_number_of_points)) {
    return -1;
  }
  PyObject *input_array = PyArray_FROM_OTF(input_obj, NPY_COMPLEX128, NPY_IN_ARRAY);
  if (input_array == NULL) {
    return -1;
  }
  self->real_map = (PyArrayObject *)input_array;

  self->ndim = PyArray_NDIM(input_array);
  int total_number_of_pixels = 1;
  int dims[self->ndim];
  int dim;
  for (dim = 0; dim < self->ndim; ++dim) {
    dims[dim] = (int) PyArray_DIM(input_array, dim);
    total_number_of_pixels *= dims[dim];
  }
  nfft_init(&self->my_plan, self->ndim, dims, self->max_number_of_points);
  self->sneaky_ref = self->my_plan.f_hat;
  self->my_plan.f_hat = (fftw_complex *)PyArray_DATA(self->real_map);
  return 0;
}

static void Transformer_dealloc(Transformer *self)
{
  Py_XDECREF(self->real_map);
  self->my_plan.f_hat = self->sneaky_ref;
  nfft_finalize(&self->my_plan);
  self->ob_type->tp_free((PyObject *)self);
}

PyDoc_STRVAR(Transformer_transform__doc__, "transform(coordinates)\n\nReturns transformation at given coordintase.\n\nCoordinate array has format [NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS] and can be of any type that can direcly be converted to an ndarray.");
static PyObject *Transformer_transform(Transformer *self, PyObject *args, PyObject *kwargs)
{
  PyObject *input_obj;
  PyObject *use_direct_obj = NULL;
  static char *kwlist[] = {"coordinates", "use_direct", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &input_obj, &use_direct_obj)) {
    return NULL;
  }
  PyObject *coordinates_array = PyArray_FROM_OTF(input_obj, NPY_FLOAT64, NPY_IN_ARRAY);
  
  if (coordinates_array == NULL) {
    return NULL;
  }

  if ((PyArray_NDIM(coordinates_array) != 2 || PyArray_DIM(coordinates_array, 1) != self->ndim) && (self->ndim != 1 || PyArray_NDIM(coordinates_array) != 1)) {
    PyErr_SetString(PyExc_ValueError, "Coordinates must be given as array of dimensions [NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS] or [NUMBER_OF_POINTS for 1D transforms.\n");
    Py_XDECREF(coordinates_array);
    return NULL;
  }

  int number_of_points = (int) PyArray_DIM(coordinates_array, 0);
  if (number_of_points > self->max_number_of_points) {
    PyErr_SetString(PyExc_ValueError, "Coordinates array is longer than max_number_of_coordinates");
    return NULL;
  }

  int use_direct = 0;
  if (use_direct_obj != NULL && PyObject_IsTrue(use_direct_obj)) {
    use_direct = 1;
  }

  //memcpy(self->my_plan.x, PyArray_DATA(coordinates_array), self->ndim*number_of_points*sizeof(double));
  double *coordinates_array_data = (double *) PyArray_DATA(coordinates_array);
  for (int i = 0; i < self->ndim*number_of_points; i++) {
    self->my_plan.x[i] = coordinates_array_data[i] * self->pixel_size;
  }

  if (self->my_plan.flags & PRE_ONE_PSI) {
    nfft_precompute_one_psi(&self->my_plan);
  }
  
  if (self->my_plan.flags & PRE_PSI) {
    nfft_precompute_psi(&self->my_plan);
  }

  if (self->my_plan.flags & PRE_FULL_PSI) {
    nfft_precompute_full_psi(&self->my_plan);
  }

  if (self->my_plan.flags & PRE_LIN_PSI) {
    nfft_precompute_lin_psi(&self->my_plan);
  }

  if (use_direct == 1) {
    nfft_trafo_direct(&self->my_plan);
  } else {
    nfft_trafo(&self->my_plan);
  }

  //npy_intp out_dim[] = {number_of_points};
  npy_intp *out_dim = malloc(1*sizeof(npy_intp));
  out_dim[0] = number_of_points;
  PyObject *out_array = (PyObject *)PyArray_SimpleNew(1, out_dim, NPY_COMPLEX128);
  memcpy(PyArray_DATA(out_array), self->my_plan.f, number_of_points*sizeof(fftw_complex));
  return out_array;
}

PyDoc_STRVAR(Transformer_ndim__doc__, "ndim()\n\nGet the number of dimensions.");
static PyObject *Transformer_ndim(Transformer *self, PyObject *args, PyObject *kwds)
{
  if (self->ndim > 0) {
    return Py_BuildValue("i", self->ndim);
  } else {
    return Py_BuildValue("");
  }
}

PyDoc_STRVAR(Transformer_max_number_of_points__doc__, "max_number_of_points()\n\nGet the maximum length of coordinates array.");
static PyObject *Transformer_max_number_of_points(Transformer *self, PyObject *args, PyObject *kwds)
{
  return Py_BuildValue("i", self->max_number_of_points);
}

static PyMethodDef Transformer_methods[] = {
  {"transform", (PyCFunction) Transformer_transform, METH_VARARGS|METH_KEYWORDS, Transformer_transform__doc__},
  {"ndim", (PyCFunction) Transformer_ndim, METH_VARARGS, Transformer_ndim__doc__},
  {"max_number_of_points", (PyCFunction) Transformer_max_number_of_points, METH_VARARGS, Transformer_max_number_of_points__doc__},
  {NULL}
};

PyDoc_STRVAR(nnfft__doc__, "nnfft(real_coordinates, real_values, fourier_coordinates)\n\nCalculate nnfft from arbitrary dimensional array.\n\real_space should be an array (or any object that can trivially be converted to one.\ncoordinates should be a NxD array where N is the number of points where the Fourier transform should be evaluated and D is the dimensionality of the input array\ndirect (optional) requires the use of the more accurate but slower ndft (default is False)");

static PyObject *nnfft(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyObject *real_coord_obj, *fourier_coord_obj;
  PyObject *real_values_obj;
  PyObject *use_direct_obj = NULL;
  
  static char *kwlist[] = {"real_coordinates", "real_values", "fourier_coordinates", "use_direct", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|O", kwlist, &real_coord_obj, &real_values_obj, &fourier_coord_obj, &use_direct_obj)) {
    return NULL;
  }
  int use_direct = 0;
  if (use_direct_obj != NULL && PyObject_IsTrue(use_direct_obj)) {
    use_direct = 1;
  }

  PyObject *real_coord_array = PyArray_FROM_OTF(real_coord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  PyObject *real_values_array = PyArray_FROM_OTF(real_values_obj, NPY_COMPLEX128, NPY_IN_ARRAY);
  PyObject *fourier_coord_array = PyArray_FROM_OTF(fourier_coord_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  if (real_coord_array == NULL || real_values_array == NULL || fourier_coord_array == NULL) {
    Py_XDECREF(real_coord_array);
    Py_XDECREF(real_values_array);
    Py_XDECREF(fourier_coord_array);
    return NULL;
  }

  int ndim = PyArray_NDIM(real_coord_array);
  if (ndim <= 0) {
    PyErr_SetString(PyExc_ValueError, "Input array can't be 0 dimensional\n");
    Py_XDECREF(real_coord_array);
    Py_XDECREF(real_values_array);
    Py_XDECREF(fourier_coord_array);
    return NULL;
  }
  
  if (ndim > 2 ||
      (PyArray_NDIM(real_coord_array) != PyArray_NDIM(fourier_coord_array)) ||
      (ndim == 2 && PyArray_DIM(real_coord_array, 1) != PyArray_DIM(fourier_coord_array, 1)) ||
      (ndim == 2 && PyArray_DIM(real_coord_array, 0) != PyArray_DIM(real_values_array, 0)) ||
      (ndim == 1 && PyArray_DIM(real_coord_array, 0) != PyArray_DIM(real_values_array, 0))) {
    PyErr_SetString(PyExc_ValueError, "Coordinates must be given as array of dimensions [NUMBER_OF_POINTS, NUMBER_OF_DIMENSIONS] or [NUMBER_OF_POINTS] for 1D transforms.\n");
    Py_XDECREF(real_coord_array);
    Py_XDECREF(real_values_array);
    Py_XDECREF(fourier_coord_array);
    return NULL;
  }

  int number_of_real_points = (int) PyArray_DIM(real_coord_array, 0);
  int number_of_fourier_points = (int) PyArray_DIM(fourier_coord_array, 0);
  int number_of_dimensions = 0;
  if (ndim == 2) {
    number_of_dimensions = (int) PyArray_DIM(fourier_coord_array, 1);
  } else {
    number_of_dimensions = 1;
  }
  
  nnfft_plan my_plan;

  int cutoff_frequencies[ndim];
  for (int dim_index = 0; dim_index < number_of_dimensions; dim_index++) {
    // Find maximum of fourier and real coordinates
    npy_intp dim_size_real = PyArray_DIM(real_coord_array, 0);
    npy_intp dim_size_fourier = PyArray_DIM(fourier_coord_array, 0);
    double max_real = 0.;
    double max_fourier = 0.;
    double *real_data = PyArray_DATA(real_coord_array);
    double *fourier_data = PyArray_DATA(fourier_coord_array);
    
    for (int i = 0; i < dim_size_real; i++)
      if (fabs(real_data[2*i + dim_index]) > max_real)
	max_real = fabs(real_data[2*i + dim_index]);
    for (int i = 0; i < dim_size_fourier; i++)
      if (fabs(fourier_data[2*i + dim_index]) > max_fourier)
	max_fourier = fabs(fourier_data[2*i + dim_index]);
    
    // Set the cut of frequency
    cutoff_frequencies[dim_index] = (int) ceil(max_real*max_fourier);
    //printf("set frequency cutoff to %d\n", cutoff_frequencies[dim_index]);
    cutoff_frequencies[dim_index] = 1;
  }

  nnfft_init(&my_plan, number_of_dimensions, number_of_fourier_points, number_of_real_points, cutoff_frequencies);
  memcpy(my_plan.x, PyArray_DATA(real_coord_array), number_of_dimensions*number_of_real_points*sizeof(double));
  memcpy(my_plan.v, PyArray_DATA(fourier_coord_array), number_of_dimensions*number_of_fourier_points*sizeof(double));

  if (my_plan.nnfft_flags &PRE_PSI) {
    nnfft_precompute_psi(&my_plan);
  }

  if (my_plan.nnfft_flags &PRE_FULL_PSI) {
    nnfft_precompute_full_psi(&my_plan);
  }

  if (my_plan.nnfft_flags &PRE_LIN_PSI) {
    nnfft_precompute_lin_psi(&my_plan);
  }

  if (my_plan.nnfft_flags &PRE_PHI_HUT) {
    nnfft_precompute_phi_hut(&my_plan);
  }

  memcpy(my_plan.f, PyArray_DATA(real_values_array), number_of_real_points*sizeof(fftw_complex));
  
  if (use_direct == 1) {
    //nnfft_trafo_direct(&my_plan);
    nnfft_adjoint_direct(&my_plan);
  } else {
    nnfft_adjoint(&my_plan);
  }

  //int out_dim[] = {number_of_fourier_points};
  npy_intp *out_dim = malloc(1*sizeof(npy_intp));
  out_dim[0] = number_of_fourier_points;

  //PyObject *out_array = (PyObject *)PyArray_FromDims(1, out_dim, NPY_COMPLEX128);
  PyObject *out_array = (PyObject *)PyArray_SimpleNew(1, out_dim, NPY_COMPLEX128);
  memcpy(PyArray_DATA(out_array), my_plan.f_hat, number_of_fourier_points*sizeof(fftw_complex));

  Py_XDECREF(real_coord_array);
  Py_XDECREF(real_values_array);
  Py_XDECREF(fourier_coord_array);

  nnfft_finalize(&my_plan);
  return out_array;
  //return Py_BuildValue("i", 5);
}

static PyTypeObject TransformerType = {
   PyObject_HEAD_INIT(NULL)
   0,                         /* ob_size */
   "Transformer",         /* tp_name */
   sizeof(Transformer),   /* tp_basicsize */
   0,                         /* tp_itemsize */
   (destructor)Transformer_dealloc, /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_compare */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags*/
   Transformer__doc__,        /* tp_doc */
   0,                         /* tp_traverse */
   0,                         /* tp_clear */
   0,                         /* tp_richcompare */
   0,                         /* tp_weaklistoffset */
   0,                         /* tp_iter */
   0,                         /* tp_iternext */
   Transformer_methods,   /* tp_methods */
   Transformer_members,   /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Transformer_init,  /* tp_init */
   0,                         /* tp_alloc */
   0,                         /* tp_new */
};

static PyMethodDef NfftMethods[] = {
  {"ndfft", (PyCFunction)ndfft, METH_VARARGS|METH_KEYWORDS, ndfft__doc__},
  {"nfft", (PyCFunction)nfft, METH_VARARGS|METH_KEYWORDS, nfft__doc__},
  {"nfft_inplace", (PyCFunction)nfft_inplace , METH_VARARGS|METH_KEYWORDS, nfft_inplace__doc__},
  {"nnfft", (PyCFunction)nnfft, METH_VARARGS|METH_KEYWORDS, nnfft__doc__},
  {NULL, NULL, 0, NULL}
};



PyMODINIT_FUNC initnfft(void)
{
  import_array();
  PyObject *m = Py_InitModule3("nfft", NfftMethods, "Nonequispaced FFT tools.");
  if (m == NULL)
    return;

  TransformerType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&TransformerType) < 0)
    return;

  Py_INCREF(&TransformerType);
  PyModule_AddObject(m, "Transformer", (PyObject *)&TransformerType);
}
